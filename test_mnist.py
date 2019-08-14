from layers import *
from functions import *
from optimizers import *
from loss_functions import *
import mnist
import pickle

from model import Model
from loss_functions import *

train_slice = slice(100)
test_slice = slice(10000,10500)
save = False
batch = True
batchSize = 5

images = mnist.train_images()
labels = mnist.train_labels()

print(images.shape[0])

train_x = images[train_slice] / 255.0
train_y = labels[train_slice] 

test_x = images[test_slice] / 255.0
test_y = labels[test_slice]

loss = CatCrossEntropy()

net = Network(optmizer=Adam(0.02))
net.add(Convolution())
net.add(MaxPool(4))
net.add(Flatten())
net.add(Dense(inputs = 288, outputs = 20, activation = LeakyReLu()))
net.add(Dense(inputs = 20, outputs = 10, activation = SoftMax()))

def accuracy(label, out):
    return 1 if np.argmax(out) == label and out[np.argmax(out)] > 0.6 else 0

def train_batch(batch):
    size = len(train_x)
    perm = np.random.permutation(size)
    x = train_x[perm]
    l = train_y[perm]
    err = 0
    acc = 0
    n = 0
    for b in range(0, size, batch):
        n += 1
        x_batch = x[b:b+batch]    
        l_batch = l[b:b+batch]
        
        e, a = net.batch_fit(x_batch, l_batch, loss, accuracy)
        err += e
        acc += a

        print("After {} steps  : error = {}, accuracy = {}".format(n, err / n , acc / n))

    return err / n, acc / n

def train(batch):
    perm = np.random.permutation(len(train_x))
    x = train_x[perm]
    l = train_y[perm]
    err = 0
    acc = 0
    n = 0
    for im, label in zip(x, l):
        n += 1
        if n > 0 and (n % batch) == 0:
            print("After {} steps  : error = {}, accuracy = {}".format(n, err / n, acc / n))
        out = net.propagate(im)

        err += loss.loss(label, out)

        dL_dO = loss.lossDerivative(label, out)
        
        acc += accuracy(label, out)

        net.backpropagate(dL_dO)

    return err / n, acc / n

def test():
    err = 0
    acc = 0
    size = len(test_x)

    for im, label in zip(test_x, test_y):
        out = net.propagate(im)

        err += loss.loss(label, out)
        acc += accuracy(label, out)

    return err / size, acc / size


cnt =  train_x.shape[0]
print("Training with {} images".format(cnt))

t = train_batch if batch else train

for epoch in range(10):
    err, acc = t(batchSize)
    print("Epoch {} : Error : {} Accuracy : {} ".format(epoch,  err, acc))
    if acc > 0.98:
        print("Training ended!!!")
        break

cnt = test_x.shape[0]
print("Testing with {} images.".format(cnt))
err, acc = test()
print("Test:  Accuracy : {} Error : {}".format(acc, err))

exp = net.export()

if save:
    print("Saving...")
    filehandler = open("./nets/net_test_2.npz","wb")
    pickle.dump(exp, filehandler)
    filehandler.close()

print("End.")



