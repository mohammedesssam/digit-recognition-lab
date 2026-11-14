"""Use original UCI writer-separated test split. Never tune on it."""
from pathlib import Path
import hashlib,json,random
import joblib,numpy as np,torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1]
SEED=42

def evaluate(y,p):return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'confusion_matrix':confusion_matrix(y,p,labels=list(range(10))).tolist()}
def main():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    raw=np.loadtxt(ROOT/'data/optdigits.tra',delimiter=',');test=np.loadtxt(ROOT/'data/optdigits.tes',delimiter=',')
    assert raw.shape==(3823,65) and test.shape==(1797,65)
    X=raw[:,:64]/16.;y=raw[:,64].astype(int);Xt=test[:,:64]/16.;yt=test[:,64].astype(int)
    assert np.isfinite(raw).all() and X.min()>=0 and X.max()<=1
    train,val=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=SEED)
    # fixed SVC baseline; no test-set hyperparameter search
    svm=SVC(C=10,gamma='scale');svm.fit(X[train],y[train]);vp=svm.predict(X[val])
    net=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10))
    opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.001)
    a=torch.tensor(X[train],dtype=torch.float32);b=torch.tensor(y[train],dtype=torch.long)
    v=torch.tensor(X[val],dtype=torch.float32);best=-1;checkpoint=None;best_epoch=None
    for epoch in range(250):
        net.train();opt.zero_grad();loss=nn.functional.cross_entropy(net(a),b);loss.backward();opt.step();net.eval()
        with torch.no_grad():pv=net(v).argmax(1).numpy()
        score=f1_score(y[val],pv,average='macro')
        if score>best:
            best=score;best_epoch=epoch+1;checkpoint={k:t.detach().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(checkpoint);net.eval()
    with torch.no_grad(): npred=net(torch.tensor(Xt,dtype=torch.float32)).argmax(1).numpy()
    spred=svm.predict(Xt)
    # Serve MLP to demonstrate actual PyTorch inference; baseline kept as comparison.
    result={'seed':SEED,'data':'Original UCI optdigits training=3823,test=1797. Separate writers in original test.',
        'split':{'train':len(train),'validation':len(val),'test':len(yt)},
        'sha256':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in ['optdigits.tra','optdigits.tes']},
        'validation':{'svm':evaluate(y[val],vp),'pytorch_mlp_macro_f1':float(best)},
        'test':{'pytorch_mlp':evaluate(yt,npred),'svm':evaluate(yt,spred)},
        'pytorch':{'epochs':250,'best_epoch':best_epoch,'selection':'Max validation macro-F1, earliest best checkpoint','layers':[64,64,10]},
        'serving':'PyTorch MLP, explicitly chosen to demonstrate neural-network inference; SVM baseline not silently substituted.',
        'limits':'8x8 block-count benchmark only, not general OCR. No evidence of real-world or current deployment accuracy.'}
    (ROOT/'reports/metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(checkpoint,ROOT/'artifacts/model.pt');joblib.dump(svm,ROOT/'artifacts/svm.joblib')
    np.savez_compressed(ROOT/'reports/splits.npz',train=train,validation=val)
    np.savez_compressed(ROOT/'reports/test_predictions.npz',actual=yt,pytorch=npred,svm=spred)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay.from_predictions(yt,npred,ax=ax,cmap='Blues',colorbar=False)
    ax.set_title('UCI writer-separated test: PyTorch MLP');fig.tight_layout();fig.savefig(ROOT/'reports/confusion_matrix.png',dpi=140)
    print(json.dumps({k:result[k] for k in ['split','pytorch','test']},indent=2))
if __name__=='__main__':main()
