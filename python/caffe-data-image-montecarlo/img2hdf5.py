import sys
import numpy
from matplotlib import pyplot
import h5py

IMAGE_SIZE = (100, 100)
MEAN_VALUE = 128

filename = sys.argv[1]
setname, ext = filename.split('.')

with open(filename, 'r') as f:
    lines = f.readlines()

numpy.random.shuffle(lines)

sample_size = len(lines)
#imgs = numpy.zeros((sample_size, 1,) + IMAGE_SIZE, dtype=numpy.float32)
imgs = numpy.zeros((sample_size, 1, 1, 9), dtype=numpy.float32)
#imgs = numpy.zeros((sample_size, 1, 1, 10000), dtype=numpy.float32)
scores = numpy.zeros(sample_size, dtype=numpy.float32)

h5_filename = '{}.h5'.format(setname)
with h5py.File(h5_filename, 'w') as h:
    for i, line in enumerate(lines):
        image_name, score = line[:-1].split()
        img = pyplot.imread(image_name)[:, :, 0].astype(numpy.float32)
        #img = img.reshape((1, )+img.shape)
        img = img.reshape(1, 1, 1, 10000)
        img = img[:,:,:,0:9]
        img -= MEAN_VALUE
        imgs[i] = img
        scores[i] = float(score)
        if (i+1) % 1000 == 0:
            print('processed {} images!'.format(i+1))
    h.create_dataset('data', data=imgs)
    h.create_dataset('score', data=scores)

with open('{}_h5.txt'.format(setname), 'w') as f:
    f.write(h5_filename)

print imgs[0]
print scores[0]
