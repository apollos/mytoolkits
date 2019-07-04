# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
import cv2
# display plots in this notebook
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.

caffe_root = '/sharefs/xiwenzh/demodata/genList/trinasolar/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
test_img = caffe_root+'test_data/Q0/Q0_A09171000600179_171006082117_H.jpg'
#test_img = '/sharefs/xiwenzh/demodata/9.27/Q2/Q2_A07170500105006.jpg'
sys.path.insert(0, caffe_root + 'python')


if os.path.isfile(caffe_root + 'model/vgg19-2_iter_10000.caffemodel.h5'):
    print 'Model found.'
else:
    exit(1)

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

model_def = caffe_root + 'model/inference2.prototxt'
model_weights = caffe_root + 'model/vgg19-2_iter_10000.caffemodel.h5'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(caffe_root + 'train_mean.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = arr[0]
mu = mean_npy.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          366,366)  # image size

image = cv2.imread(test_img)
image=cv2.resize(image,(366,366),0,0,interpolation=cv2.INTER_LINEAR)

blob = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
#blob[0, :] = (image - mu)
blob[0, :] = image 
channel_swap = (0, 3, 1, 2)
blob = blob.transpose(channel_swap)
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = blob - arr

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = caffe_root + 'synsets.txt'
if not os.path.exists(labels_file):
    print("Can not find synsets file")
    exit(1)

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:3]  # reverse sort and take five largest items

print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])
