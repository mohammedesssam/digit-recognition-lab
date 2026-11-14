from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.api import app
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def client():
    with TestClient(app) as c:yield c

def test_source_and_splits():
    m=json.loads((ROOT/'reports/metrics.json').read_text())
    for n,h in m['sha256'].items():assert hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()==h
    s=np.load(ROOT/'reports/splits.npz');assert not set(s['train'])&set(s['validation'])
    assert len(s['train'])+len(s['validation'])==3823

def test_metric_reproduction():
    from sklearn.metrics import accuracy_score,f1_score
    d=np.load(ROOT/'reports/test_predictions.npz');m=json.loads((ROOT/'reports/metrics.json').read_text())['test']['pytorch_mlp']
    assert accuracy_score(d['actual'],d['pytorch'])==m['accuracy']
    assert f1_score(d['actual'],d['pytorch'],average='macro')==m['macro_f1']

def test_http(client):
    assert client.get('/health').json()['model']=='pytorch_mlp'
    assert client.get('/').status_code==200 and client.get('/docs').status_code==200
    d=client.get('/example/0').json();a=client.post('/predict',json={'pixels':d['pixels']})
    assert a.status_code==200 and 0<=a.json()['digit']<=9
    assert abs(sum(x['softmax_score'] for x in a.json()['top3']))<=1.000001
    assert a.json()==client.post('/predict',json={'pixels':d['pixels']}).json()

@pytest.mark.parametrize('data',[{}, {'pixels':[0]*63},{'pixels':[0]*65},{'pixels':[17]*64},{'pixels':[-1]*64},{'pixels':[0]*64},{'pixels':[1]*64,'label':1}])
def test_bad_input(client,data):assert client.post('/predict',json=data).status_code==422

def test_missing_example(client):assert client.get('/example/9999').status_code==404

def test_online_matches_all_saved_predictions(client):
    # Batched checkpoint reload independently reproduces every held-out prediction.
    import torch
    from app.api import state
    d=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',')
    with torch.no_grad():p=state['model'](torch.tensor(d[:,:64]/16,dtype=torch.float32)).argmax(1).numpy()
    assert np.array_equal(p,np.load(ROOT/'reports/test_predictions.npz')['pytorch'])
