import pickle
from layers import *
import mnist
import pickle

images = mnist.train_images()
labels = mnist.train_labels()

test_x = images[15000:18000] / 255.0
test_y = labels[15000:18000]

filehandler = open("./nets/net_test_5k.npz","rb")
o = pickle.load(filehandler)
filehandler.close()

net = Network.create(o)

def error(out, label):
    return -np.log(out[label] + 1e-8)

def accuracy(out, label):
    return 1 if np.argmax(out) == label and  out[np.argmax(out)] > 0.6 else 0

def test():
    err = 0
    acc = 0

    for i, (im, label) in enumerate(zip(test_x, test_y)):
        out = net.propagate(im)

        err += error(out, label)
        acc += accuracy(out, label)

    return err, acc

cnt = test_x.shape[0]

print("Testing with {} images.".format(cnt))
err, acc = test()
err /= cnt
acc /= cnt
print("Test:  Accuracy : {} Error : {}".format(acc, err))

