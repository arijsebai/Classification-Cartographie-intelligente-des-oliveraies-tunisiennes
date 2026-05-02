import requests, json
r = requests.get('http://127.0.0.1:8000/api/parcels')
print('status', r.status_code)
js = r.json()
print('features', len(js.get('features', [])))
if js.get('features'):
    feat = js['features'][0]
    props = feat.get('properties', {})
    print('properties keys:', list(props.keys()))
    sample = {k: props.get(k) for k in ['id', 'name', 'cultivation_system', 'model_prediction', 'prob_intensif']}
    print(json.dumps(sample, indent=2, ensure_ascii=False))
else:
    print('no features')
