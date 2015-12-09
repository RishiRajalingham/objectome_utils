import cPickle as pk
import caffe
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

""" This is a slightly edited version of Shay's OM script."""

caffe_root = '/om/user/shayo/caffe/caffe/';

# stimpath = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
stimpath = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100nobg/'
cnn_oi = 'caffe_reference'
desired_layer = 'fc8'


# Get image files
meta = pk.load(open(stimpath + 'metadata.pkl'))
filelist = [stimpath + 'images/' + m['id'] + '.png' for m in meta]
print("There are "+str(len(filelist))+ " files")

# Set Model Parameters
if cnn_oi == 'caffe_reference':
    input_size = 227
    dim = 1000
    proto_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_file = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    model_input = '/om/user/shayo/caffe/caffe/models/bvlc_reference_caffenet'
    output_path = stimpath + 'caffe_features/'
elif cnn_oi == 'VGG_S':
    input_size = 224
    dim = 1000
    proto_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S_deploy.prototxt'
    model_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S.caffemodel'
    output_file = stimpath + 'vgg_features/'

sys.path.insert(0, caffe_root + 'python')
model_call = '/om/user/shayo/caffe/caffe/scripts/download_model_binary.py ' + model_input
if not os.path.isfile(model_file):
    print("Model file is missing")
    get_ipython().system(model_call)

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap']='gray'
caffe.set_mode_cpu()
net = caffe.Net(proto_file, model_file, caffe.TEST)

# Preprocess input data
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Run Model
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

# Save output
if os.path.exists(output_path) == False:
    os.mkdir(output_path)
with open(output_path + desired_layer + '.pkl', 'wb') as _f:
    pk.dump(features_dict, _f)

# features_array = np.array(features_array)
# np.save(output_file, features)
# np.save(output_file+'raw', features_array)

