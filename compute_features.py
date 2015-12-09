import cPickle as pk
import caffe
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

""" This is just a rewrite of Shay's script."""

caffe_root = '/om/user/shayo/caffe/caffe/';
stimpath = '/mindhive/dicarlolab/common/shared_stimuli/rishi_mwk/'
featurepath = '/mindhive/dicarlolab/common/shared_features/'

imgset = 'obj64s100/'
cnn_oi = 'caffe_reference'
desired_layer = 'fc8'
input_path = stimpath + imgset
output_file = featurepath + imgset + cnn_oi + desired_layer


""" Get image files """
meta = pk.load(open(input_path + 'metadata.pkl'))
filelist = [input_path + m['id'] + '.png' for m in meta]
print("There are "+str(len(filelist))+ " files")

""" Set Model Parameters """
if cnn_oi == 'caffe_reference':
    input_size = 227
    dim = 1000
    proto_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_file = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
elif cnn_oi == 'VGG_S':
    input_size = 224
    dim = 1000
    proto_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S_deploy.prototxt'
    model_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S.caffemodel'

sys.path.insert(0, caffe_root+'python')
if not os.path.isfile(model_file):
    print("Model file is missing")
    get_ipython().system(u'/om/user/shayo/caffe/caffe/scripts/download_model_binary.py /om/user/shayo/caffe/caffe/models/bvlc_reference_caffenet')

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap']='gray'
caffe.set_mode_cpu()
net = caffe.Net(proto_file, model_file, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# """ Run model once """

# # run the network just to findout the size
# # set net to batch size of 50

# net.blobs['data'].reshape(2,3,input_size,input_size)
# net.blobs['data'].data[...] =  transformer.preprocess('data',caffe.io.load_image(filelist[0]))

# out = net.forward()
# layerSize = [(k, v.data.shape) for k, v in net.blobs.items()]

# feat = net.blobs[desiredLayer].data[0]
# dim=feat.shape[0]
""" Run Model """
batchsize = 50
net.blobs['data'].reshape(batchsize,3,input_size,input_size)
indices = np.arange(0,1+len(filelist),batchsize)
features = np.zeros([len(filelist), dim])

for i in range(0,len(indices)-1):
    print "Processing "+str(indices[i])+" to "+ str(indices[i+1])
    images = filelist[indices[i]:indices[i+1]]
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()
    outdata = net.blobs[desired_layer].data
    features[indices[i]:indices[i+1],:] = outdata

    #[(k, v.data.shape) for k, v in net.blobs.items()]
    # now, assign features
    # features_array.append(outdata)
    # for j in range(0,batchsize): #range(indices[i],indices[i+1]+1):
        # features_dict[indices[i]+j]['feature'] = outdata[j,:]
 
features_dict = []
for i,m in enumerate(meta):
    f_ = {'id':m['id'], 
        'feature':features[i,:], 
        'feature_layer':desired_layer, 
        'model_id':cnn_oi }
    features_dict.append(f_)

with open(output_file + '.pkl', 'wb') as _f:
    pk.dump(features_dict, _f)

# features_array = np.array(features_array)
# np.save(output_file, features)
# np.save(output_file+'raw', features_array)

