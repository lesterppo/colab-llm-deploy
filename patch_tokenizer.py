import json
p = '/content/gemma4_local/tokenizer_config.json'
tc = json.load(open(p))
print('before extra_special_tokens:', tc.get('extra_special_tokens'))
# transformers 4.57 GemmaFast expects extra_special_tokens to be a dict
# (it calls .keys()); Gemma-4 ships it as a list ['<|video|>'].
# The video token itself lives in tokenizer.json, so dropping the list
# entry does not remove the token from the vocabulary; it only fixes
# the config shape. Weights are untouched.
tc.pop('extra_special_tokens', None)
json.dump(tc, open(p, 'w'), indent=2)
print('patched: extra_special_tokens removed')
