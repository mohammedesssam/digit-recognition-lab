from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import json,numpy as np,torch
from torch import nn
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
ROOT=Path(__file__).resolve().parents[1];state={}
@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(1)
    model=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10))
    model.load_state_dict(torch.load(ROOT/'artifacts/model.pt',weights_only=True,map_location='cpu'));model.eval()
    state['model']=model;state['examples']=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',');yield;state.clear()
app=FastAPI(title='Digit Recognition Lab',lifespan=lifespan)
class Input(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    pixels:list[Annotated[int,Field(ge=0,le=16)]]=Field(min_length=64,max_length=64)
@app.get('/')
def ui():return FileResponse(ROOT/'static/index.html')
@app.get('/health')
def health():return {'status':'ok','model':'pytorch_mlp'}
@app.get('/metrics')
def metrics():return json.loads((ROOT/'reports/metrics.json').read_text())
@app.get('/example/{index}')
def example(index:int):
    if not 0<=index<len(state['examples']):raise HTTPException(404,'No such example')
    row=state['examples'][index];return {'pixels':row[:64].astype(int).tolist(),'actual_digit':int(row[64])}
@app.post('/predict')
def predict(d:Input):
    if max(d.pixels)==0:raise HTTPException(422,'Blank input is not a digit')
    x=torch.tensor([d.pixels],dtype=torch.float32)/16
    with torch.no_grad():scores=torch.softmax(state['model'](x),dim=1).numpy()[0]
    top=np.argsort(scores)[-3:][::-1]
    return {'digit':int(top[0]),'top3':[{'digit':int(k),'softmax_score':float(scores[k])} for k in top],
        'warning':'Softmax scores are not calibrated confidence. This is an 8x8 UCI benchmark demo, not general OCR.'}
