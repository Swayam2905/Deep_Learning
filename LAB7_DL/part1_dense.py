import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def drelu(z):
    return (z>0).astype(float)

def dsigmoid(a):
    return a*(1-a)

def bce(y,yh):
    eps=1e-8
    yh=np.clip(yh,eps,1-eps)
    return -np.mean(y*np.log(yh)+(1-y)*np.log(1-yh))

def acc(y,yh):
    return np.mean((yh>=0.5)==y)

def generate_data(N=3000):
    x1=np.random.uniform(-2,2,N)
    x2=np.random.uniform(-2,2,N)
    X=np.vstack([x1,x2]).T
    y=((x1**2+x2**2)>1.5).astype(int)
    return X,y.reshape(-1,1)

def split(X,y):
    N=len(X)
    idx=np.random.permutation(N)
    X,y=X[idx],y[idx]
    t=int(0.7*N)
    v=int(0.85*N)
    return X[:t],y[:t],X[t:v],y[t:v],X[v:],y[v:]

def count_params(layers):
    total=0
    for i in range(len(layers)-1):
        total+=layers[i]*layers[i+1]+layers[i+1]
    return total

class DenseNet:
    def __init__(self,layers,activation="relu"):
        self.layers=layers
        self.activation=activation
        self.params={}
        for i in range(len(layers)-1):
            self.params["W"+str(i)]=np.random.randn(layers[i],layers[i+1])*0.1
            self.params["b"+str(i)]=np.zeros((1,layers[i+1]))

    def forward(self,X):
        self.cache={"A0":X}
        A=X
        for i in range(len(self.layers)-2):
            Z=A@self.params["W"+str(i)]+self.params["b"+str(i)]
            A=relu(Z) if self.activation=="relu" else sigmoid(Z)
            self.cache["Z"+str(i+1)]=Z
            self.cache["A"+str(i+1)]=A
        Z=A@self.params["W"+str(len(self.layers)-2)]+self.params["b"+str(len(self.layers)-2)]
        A=sigmoid(Z)
        self.cache["A"+str(len(self.layers)-1)]=A
        return A

    def backward(self,y):
        grads={}
        m=y.shape[0]
        L=len(self.layers)-1
        dZ=self.cache["A"+str(L)]-y
        for i in reversed(range(L)):
            A_prev=self.cache["A"+str(i)]
            grads["W"+str(i)]=A_prev.T@dZ/m
            grads["b"+str(i)]=np.sum(dZ,axis=0,keepdims=True)/m
            if i>0:
                dA=dZ@self.params["W"+str(i)].T
                if self.activation=="relu":
                    dZ=dA*drelu(self.cache["Z"+str(i)])
                else:
                    dZ=dA*dsigmoid(self.cache["A"+str(i)])
        return grads

    def update(self,grads,lr):
        for k in grads:
            self.params[k]-=lr*grads[k]

import matplotlib.pyplot as plt
import os

def train_dense(layers, activation="relu", epochs=200, lr=0.01):

    if not os.path.exists("plots"):
        os.makedirs("plots")

    X,y = generate_data()
    Xtr,ytr,Xval,yval,Xte,yte = split(X,y)

    model = DenseNet(layers, activation)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    grad_norms = []

    for epoch in range(epochs):

        yh = model.forward(Xtr)
        loss = bce(ytr, yh)
        grads = model.backward(ytr)
        model.update(grads, lr)

        # store gradient norm (first layer)
        grad_norms.append(np.linalg.norm(grads["W0"]))

        # validation
        val_pred = model.forward(Xval)

        train_losses.append(loss)
        val_losses.append(bce(yval, val_pred))
        train_accs.append(acc(ytr, yh))
        val_accs.append(acc(yval, val_pred))

    # Save plots
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig("plots/dense_loss.png")
    plt.close()

    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.savefig("plots/dense_accuracy.png")
    plt.close()

    plt.plot(grad_norms)
    plt.title("Gradient Norm (Layer 1)")
    plt.savefig("plots/dense_gradient_norm.png")
    plt.close()

    train_acc = train_accs[-1]
    val_acc = val_accs[-1]
    test_acc = acc(yte, model.forward(Xte))

    return train_acc, val_acc, test_acc, count_params(layers)