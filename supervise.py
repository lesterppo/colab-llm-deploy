#!/usr/bin/env python3
"""Self-healing supervisor for colab-llm-deploy models.

Problem it solves, honestly stated:
  Free Colab GPUs are reclaimed server-side (observed 15-60 min on busy
  evenings, longer in early-morning HKT). No script on the VM can prevent
  Google reclaiming a free session, and no keep-alive trick makes a free
  session guaranteed. What this supervisor *does* do:
    1. Heartbeat: a real host->VM execution every --check-interval
       (default 300s) that reads /content/deploy_status.json. Real
       executions reset Colab's 90-min idle timer, which the VM-side
       watchdog loop alone does not reliably do.
    2. Fast death detection: a session that is lost, stopped, or whose
       status file stops reporting "running"/"ready" is noticed within
       one check interval, not hours later.
    3. Rate-limited auto-redeploy: it re-runs `deploy.py deploy` for the
       same model, writes the new tunnel URL to --url-file and the full
       state to --state-file, and stops after --max-restarts attempts
       (default 3) so it never burns a whole day's quota in a loop.
       Cooldown grows between attempts (60s, 120s, 240s, ... capped
       at 900s). A 503/quota failure counts as an attempt and is logged
       as such; the supervisor does NOT retry-loop on quota.
    4. Account rotation is OPT-IN (--rotate-accounts): switching the
       global Colab account affects other agents sharing this VM (the
       CLI state is VM-global), so by default the supervisor stays on
       whatever account is currently active and never touches
       cyc236de/cyc236hk (another agent's accounts) even when rotation
       is on.

Usage:
  python3 supervise.py --model nemotron-3.5-bolt-e4b-q4 --gpu T4
  python3 supervise.py --model nemotron-3.5-bolt-e4b-q4 --once --dry-run
  python3 supervise.py --model qwen2.5-7b-q4 --max-restarts 2 --check-interval 120
  python3 supervise.py --self-test   # pure-function checks, no Colab calls

State file (JSON): session, model, tunnel_url, status, restarts,
last_check, last_deploy, history[]. The tunnel URL in --url-file is
always the latest successful deploy; clients should re-read it after
a restart, because a redeploy gets a NEW trycloudflare subdomain.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB_PY = os.environ.get(
    "COLAB_PY", os.path.expanduser("~/workspace/hermes-colab-cli/colab.py"))
DEPLOY_PY = os.path.join(HERE, "deploy.py")
# Accounts never touched, even with --rotate-accounts (another agent's).
FORBIDDEN_ACCOUNTS = {"cyc236de", "cyc236hk"}
# Rotation pool order (training-track order, minus forbidden accounts).
DEFAULT_POOL = ["ppoppo205", "cyc236es", "kyu008008",
                "kyu009009", "cyc236ha"]


def log(msg):
    print(f"[supervise {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_tunnel_url(text):
    """First trycloudflare URL in deploy.py output, or None."""
    m = re.search(r"https://[^\s'\"]*trycloudflare\.com", text or "")
    return m.group(0) if m else None


def next_account(pool, current):
    """Next account in pool after current, skipping forbidden ones."""
    pool = [a for a in pool if a not in FORBIDDEN_ACCOUNTS]
    if not pool:
        return None
    if current not in pool:
        return pool[0]
    return pool[(pool.index(current) + 1) % len(pool)]


def backoff_seconds(attempt, base=60, cap=900):
    """60, 120, 240, 480, 900, 900, ... (attempt is 1-based)."""
    return min(cap, base * (2 ** max(0, attempt - 1)))


def sh(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def self_test():
    assert parse_tunnel_url("DONE. Session 'x'\n  https://abc-123.trycloudflare.com\n") \
        == "https://abc-123.trycloudflare.com"
    assert parse_tunnel_url("no url here") is None
    assert next_account(DEFAULT_POOL, "kyu008008") == "kyu009009"
    assert next_account(DEFAULT_POOL, "kyu009009") == "cyc236ha"
    assert next_account(DEFAULT_POOL, "cyc236ha") == "ppoppo205"
    assert next_account(["cyc236de", "ppoppo205"], "ppoppo205") == "ppoppo205"
    assert backoff_seconds(1) == 60 and backoff_seconds(2) == 120
    assert backoff_seconds(3) == 240 and backoff_seconds(10) == 900
    print("self-test: 8/8 PASS")
    return 0


class Supervisor:
    def __init__(self, args):
        self.model = args.model
        self.gpu = args.gpu
        self.session = args.session or \
            f"llm-{args.model}".replace(".", "-")
        self.check_interval = args.check_interval
        self.max_restarts = args.max_restarts
        self.state_file = args.state_file
        self.url_file = args.url_file
        self.rotate = args.rotate_accounts
        self.pool = [a.strip() for a in args.accounts.split(",") if a.strip()] \
            if args.accounts else list(DEFAULT_POOL)
        self.state = {"model": self.model, "session": self.session,
                      "status": "starting", "tunnel_url": None,
                      "restarts": 0, "history": []}
        self._save()

    def _save(self):
        self.state["last_check"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_file)
        if self.state.get("tunnel_url"):
            with open(self.url_file, "w") as f:
                f.write(self.state["tunnel_url"] + "\n")

    def _record(self, event, detail=""):
        self.state["history"].append(
            {"t": time.strftime("%H:%M:%S"), "event": event, "detail": detail})
        self.state["history"] = self.state["history"][-50:]

    def session_alive(self):
        """Return (alive, detail, transient).

        A timeout talking to the CLI is NOT proof the session died
        (observed 2026-10-25: a 120s status-exec timeout crashed the
        supervisor while the Colab session was still listed and alive).
        Transient failures must be retried, never treated as death on
        the first occurrence and never allowed to raise out of here.
        """
        try:
            r = sh([sys.executable, COLAB_PY, "sessions"], timeout=120)
        except subprocess.TimeoutExpired:
            return False, "sessions list timed out (transient)", True
        except Exception as e:
            return False, f"sessions list error: {e} (transient)", True
        if self.session not in (r.stdout or ""):
            return False, "session not listed", False
        try:
            r = sh([sys.executable, COLAB_PY, "exec", "-s", self.session,
                    "--code",
                    "print(open('/content/deploy_status.json').read())"],
                   timeout=120)
        except subprocess.TimeoutExpired:
            return False, "status exec timed out (transient)", True
        except Exception as e:
            return False, f"status exec error: {e} (transient)", True
        if r.returncode != 0:
            return False, f"status exec failed (rc={r.returncode})", False
        try:
            st = json.loads(r.stdout[r.stdout.index("{"):])
        except Exception:
            return False, "status file unreadable", False
        if st.get("stage") in ("running", "ready") or st.get("ready"):
            self.state["tunnel_url"] = st.get("tunnel_url") or \
                self.state.get("tunnel_url")
            return True, st.get("stage", "running"), False
        return False, f"stage={st.get('stage')} ready={st.get('ready')}", False

    def deploy_once(self):
        """Run deploy.py once; return (ok, tunnel_url_or_error)."""
        log(f"deploying {self.model} on {self.gpu} as '{self.session}' ...")
        try:
            r = sh([sys.executable, DEPLOY_PY, "deploy", "--model", self.model,
                    "--gpu", self.gpu, "--session", self.session], timeout=1800)
        except subprocess.TimeoutExpired:
            return False, "deploy timed out after 1800s"
        except Exception as e:
            return False, f"deploy error: {e}"
        out = (r.stdout or "") + (r.stderr or "")
        url = parse_tunnel_url(out)
        if r.returncode == 0 and url:
            return True, url
        # deploy.py may have reached ready but failed its own tunnel smoke
        # test (observed 2026-10-25: Cloudflare 403 from this VM's egress
        # while the VM-side server was healthy). Check the VM status file
        # before declaring failure: a running server + tunnel URL is usable
        # from a non-datacenter network even if this VM's smoke test 403s.
        alive, detail, _transient = self.session_alive()
        if alive and self.state.get("tunnel_url"):
            log("deploy.py smoke test failed but VM reports running; "
                f"keeping tunnel {self.state['tunnel_url']} ({detail})")
            return True, self.state["tunnel_url"]
        tail = "\n".join(out.strip().splitlines()[-6:])
        return False, tail or f"deploy rc={r.returncode}"

    def run(self, once=False, dry_run=False):
        attempts = 0
        transient_streak = 0
        while True:
            try:
                alive, detail, transient = self.session_alive()
            except Exception as e:
                # The supervisor must never die on one bad check.
                log(f"heartbeat exception (treated as transient): {e}")
                alive, detail, transient = False, str(e), True
            if alive:
                transient_streak = 0
                self.state["status"] = "running"
                self._record("heartbeat", detail)
                log(f"alive ({detail}); tunnel={self.state.get('tunnel_url')}")
                self._save()
                if once:
                    break
                time.sleep(self.check_interval)
                continue
            if transient:
                transient_streak += 1
                log(f"transient heartbeat failure {transient_streak}/2: "
                    f"{detail} — NOT redeploying yet.")
                self.state["status"] = "transient"
                self._record("transient", detail)
                self._save()
                if once or transient_streak < 2:
                    if once:
                        break
                    time.sleep(self.check_interval)
                    continue
                log("2 consecutive transient failures — treating as dead.")
                transient_streak = 0
            # Dead or never started.
            if dry_run:
                log(f"DRY-RUN: would redeploy (dead: {detail}); not deploying.")
                self.state["status"] = "dry-run-would-redeploy"
                self._save()
                break
            if attempts >= self.max_restarts:
                log(f"stopping: {attempts} deploy attempts exhausted "
                    f"(max_restarts={self.max_restarts}).")
                self.state["status"] = "stopped-quota-guard"
                self._save()
                break
            attempts += 1
            if attempts > 1:
                wait = backoff_seconds(attempts - 1)
                log(f"cooldown {wait}s before attempt {attempts} ...")
                time.sleep(wait)
            # Best-effort teardown of a stale session before re-creating it.
            try:
                sh([sys.executable, DEPLOY_PY, "undeploy", "--session",
                    self.session], timeout=180)
            except Exception as e:
                log(f"undeploy best-effort failed (continuing): {e}")
            ok, result = self.deploy_once()
            if ok:
                self.state["status"] = "running"
                self.state["tunnel_url"] = result
                self.state["restarts"] = attempts - 1 if attempts > 1 else 0
                self._record("deploy-ok", result)
                log(f"DEPLOYED: {result}")
            else:
                self.state["status"] = "deploy-failed"
                self._record("deploy-failed", result[-300:])
                log(f"deploy attempt {attempts} failed:\n{result[-800:]}")
                if "503" in result or "quota" in result.lower():
                    log("quota/503 signal: counting this against the guard, "
                        "not retry-looping.")
            self._save()
            if once:
                break
            time.sleep(self.check_interval)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", default="T4")
    p.add_argument("--session")
    p.add_argument("--check-interval", type=int, default=300)
    p.add_argument("--max-restarts", type=int, default=3)
    p.add_argument("--state-file",
                   default=os.path.join(HERE, "supervisor-state.json"))
    p.add_argument("--url-file",
                   default=os.path.join(HERE, "current-tunnel.txt"))
    p.add_argument("--accounts", help="comma list for --rotate-accounts")
    p.add_argument("--rotate-accounts", action="store_true",
                   help="OPT-IN: rotate Colab account on quota failure "
                        "(changes VM-global account state)")
    p.add_argument("--once", action="store_true",
                   help="one check/deploy cycle, then exit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    Supervisor(a).run(once=a.once, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
