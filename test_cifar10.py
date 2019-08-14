import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadCifar(batch = 'data_batch_1'):
    d = unpickle("./cifar/cifar-10-batches-py/{}".format(batch))
    return d

def getRGB(img , sizeX=32, sizeY=32):
    return np.asarray(img).reshape(3, sizeY, sizeX).transpose([1, 2, 0])

def transformCifar(d : dict):
    data = []
    labels = d[b'labels'][:10]
    for img in d[b'data'][:10]:
        data.append(getRGB(img))
    return data, labels


c = loadCifar()

images, labels = transformCifar(c)

from matplotlib import pyplot as PLT

for i in range(10):
    PLT.imshow(images[i])
    PLT.show()