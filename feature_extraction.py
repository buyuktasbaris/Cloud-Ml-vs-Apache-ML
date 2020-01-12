
import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2



#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227



def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/baris/Desktop/mean.binaryproto', 'rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

caffe.set_mode_gpu()
caffe.set_device(0)
#Read model architecture and trained model's weights
net = caffe.Net('/home/baris/caffe/caffe_models/caffe_model_2/adil_deploy.prototxt',
                '/media/baris/Data/caffe_model_444_iter_80000.caffemodel',
                caffe.TEST)
#net = caffe.Net('/home/baris/caffe/caffe_models/caffe_model_2/adil_deploy.prototxt',
 #               '/home/baris/caffe/input/bvlc_reference_caffenet.caffemodel',
  #              caffe.TEST)
#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))



file=open('/home/baris/caffe/input/casia_test.txt', 'r')


labels=[]

#Reading image paths
test_img_paths = []

#file=open('/home/baris/caffe/input/casia_test.txt', 'r')
for in_idx,line in enumerate(file.readlines()): 
		test_img_paths.append(line)
#Making predictions
all_data=np.zeros((len(test_img_paths),4096))
count=0
for in_idx,img_path in enumerate(test_img_paths):
    a=img_path.split(' ')
    
    labels.append(int(a[1]))
    img = cv2.imread(a[0], cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    count=count+1
    print(count)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    caffe_ft = net.blobs['fc7_sun2'].data[0]
    all_data[in_idx,:]=caffe_ft
np.save('/media/baris/Data/curriculum/casia_test.npy', all_data)
np.save('/media/baris/Data/curriculum/casia_test_labels.npy', labels)



file=open('/home/baris/caffe/input/casia_train.txt', 'r')
labels=[]

#Reading image paths
test_img_paths = []

#file=open('/home/baris/caffe/deeplearning-cats-dogs-tutorial/input/casia_test.txt', 'r')
for in_idx,line in enumerate(file.readlines()): 
		test_img_paths.append(line)
all_data=np.zeros((len(test_img_paths),4096))
#Making predictions
count=0
for in_idx,img_path in enumerate(test_img_paths):
    a=img_path.split(' ')
    
    labels.append(int(a[1]))
    img = cv2.imread(a[0], cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    count=count+1
    print(count)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    caffe_ft = net.blobs['fc7_sun2'].data[0]
    all_data[in_idx,:]=caffe_ft
np.save('/media/baris/Data/curriculum/casia_train.npy', all_data)
np.save('/media/baris/Data/curriculum/casia_train_labels.npy', labels)