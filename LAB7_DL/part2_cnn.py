import numpy as np
import matplotlib.pyplot as plt
import os

def relu(z): return np.maximum(0,z)
def sigmoid(z): return 1/(1+np.exp(-z))
def bce(y,yh):
    eps=1e-8
    yh=np.clip(yh,eps,1-eps)
    return -np.mean(y*np.log(yh)+(1-y)*np.log(1-yh))
def acc(y,yh): return np.mean((yh>=0.5)==y)

def generate_images(N=3000):
    X=[]; y=[]
    for _ in range(N):
        img=np.zeros((8,8))
        if np.random.rand()>0.5:
            img[:,4]=1; label=0
        else:
            img[4,:]=1; label=1
        img+=np.random.normal(0,0.1,(8,8))
        X.append(img); y.append(label)
    return np.array(X),np.array(y).reshape(-1,1)

def split(X,y):
    N=len(X)
    idx=np.random.permutation(N)
    X,y=X[idx],y[idx]
    t=int(0.7*N); v=int(0.85*N)
    return X[:t],y[:t],X[t:v],y[t:v],X[v:],y[v:]

class CNN:
    def __init__(self):
        self.K=np.random.randn(3,3)*0.1
        self.W=np.random.randn(36,1)*0.1
        self.b=np.zeros((1,1))

    def conv(self,X):
        out=np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                out[i,j]=np.sum(X[i:i+3,j:j+3]*self.K)
        return out

    def forward(self,X):
        c=self.conv(X)
        r=relu(c)
        self.flat=r.flatten().reshape(1,-1)
        return sigmoid(self.flat@self.W+self.b)

    def backward(self,y,yh,lr):
        dZ=yh-y
        dW=self.flat.T@dZ
        self.W-=lr*dW



def train_cnn(epochs=5, lr=0.01):

    if not os.path.exists("plots"):
        os.makedirs("plots")

    X,y = generate_images()
    Xtr,ytr,Xval,yval,Xte,yte = split(X,y)

    model = CNN()

    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        for i in range(len(Xtr)):
            yh = model.forward(Xtr[i])
            model.backward(ytr[i], yh, lr)

        train_pred = np.array([model.forward(x) for x in Xtr])
        val_pred = np.array([model.forward(x) for x in Xval])

        train_accs.append(acc(ytr, train_pred))
        val_accs.append(acc(yval, val_pred))

    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("CNN Accuracy vs Epoch")
    plt.legend()
    plt.savefig("plots/cnn_accuracy.png")
    plt.close()

    test_acc = acc(yte, np.array([model.forward(x) for x in Xte]))
    params = 3*3 + 1 + 36 + 1

    return train_accs[-1], val_accs[-1], test_acc, params