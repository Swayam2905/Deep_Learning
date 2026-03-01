import numpy as np
from part2_cnn import generate_images, split, CNN, acc
import matplotlib.pyplot as plt

def train_optimizer(opt="sgd", epochs=5, lr=0.01):

    X,y = generate_images()
    Xtr,ytr,Xval,yval,Xte,yte = split(X,y)

    model = CNN()

    train_losses = []

    vW = np.zeros_like(model.W)
    mW = np.zeros_like(model.W)
    vW2 = np.zeros_like(model.W)
    t = 0

    for epoch in range(epochs):

        total_loss = 0

        for i in range(len(Xtr)):
            yh = model.forward(Xtr[i])
            dZ = yh - ytr[i]
            dW = model.flat.T @ dZ

            total_loss += np.sum(dZ**2)

            if opt == "sgd":
                model.W -= lr*dW

            elif opt == "momentum":
                vW = 0.9*vW + lr*dW
                model.W -= vW

            elif opt == "adam":
                t += 1
                mW = 0.9*mW + 0.1*dW
                vW2 = 0.999*vW2 + 0.001*(dW**2)
                mhat = mW/(1-0.9**t)
                vhat = vW2/(1-0.999**t)
                model.W -= lr*mhat/(np.sqrt(vhat)+1e-8)

        train_losses.append(total_loss/len(Xtr))

    train_acc = acc(ytr, np.array([model.forward(x) for x in Xtr]))
    val_acc = acc(yval, np.array([model.forward(x) for x in Xval]))
    test_acc = acc(yte, np.array([model.forward(x) for x in Xte]))
    params = 3*3 + 1 + 36 + 1

    return train_acc, val_acc, test_acc, params, train_losses