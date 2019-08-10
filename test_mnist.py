from layers import *
from functions import *
from optimizers import *
import mnist
import pickle

from model import Model
from loss_functions import *

images = mnist.train_images()
labels = mnist.train_labels()

print(images.shape[0])

train_x = images[:1000] / 255.0
train_y = labels[:1000] 

test_x = images[10000:10500] / 255.0
test_y = labels[10000:10500]


net = Network(optmizer=SGDMomentum(0.02, 0.9))
net.add(Convolution())
net.add(MaxPool())
net.add(Flatten())
net.add(Dense(inputs = 1352, outputs = 20, activation = LeakyReLu()))
net.add(Dense(inputs = 20, outputs = 10, activation = SoftMax()))

def error(out, label):
    return -np.log(out[label] + 1e-8)

def errorDerivative(out, label):
    dL_dO = np.zeros(10)
    dL_dO[label] = 1 / (out[label] + 1e-8)
    return dL_dO

def accuracy(out, label):
    return 1 if np.argmax(out) == label and  out[np.argmax(out)] > 0.6 else 0

def train():
    perm = np.random.permutation(len(train_x))
    x = train_x[perm]
    l = train_y[perm]
    err = 0
    acc = 0
    update = train_x.shape[0] / 100
    for i, (im, label) in enumerate(zip(x, l)):
        if i > 0 and (i % update) == update - 1:
            print("After {} steps  : error = {}, accuracy = {}".format(i+1, err / (i+1), acc / (i+1)))
        out = net.propagate(im)

        err += error(out, label)

        dL_dO = errorDerivative(out, label)
        
        acc += accuracy(out, label)

        net.backpropagate(dL_dO)

    return err, acc

def test():
    err = 0
    acc = 0

    for i, (im, label) in enumerate(zip(test_x, test_y)):
        out = net.propagate(im)

        err += error(out, label)
        acc += accuracy(out, label)

    return err, acc


cnt =  train_x.shape[0]
print("Training with {} images".format(cnt))

for epoch in range(10):
    err, acc = train()
    err /= cnt
    acc /= cnt
    print("Epoch {} : Accuracy : {} Error : {}".format(epoch,  acc, err))
    if acc > 0.98:
        print("Training ended!!!")
        break

"""m = Model(net, CatCrossEntropy(), accuracy)

err, acc = m.train(train_x, train_y, 10)"""

cnt = test_x.shape[0]
print("Testing with {} images.".format(cnt))
err, acc = test()
err /= cnt
acc /= cnt
print("Test:  Accuracy : {} Error : {}".format(acc, err))

exp = net.export()

print("Saving...")

filehandler = open("./nets/net_test__.npz","wb")
pickle.dump(exp, filehandler)
filehandler.close()

print("End.")



