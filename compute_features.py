import cPickle as pk
import caffe
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

""" This is a slightly edited version of Shay's OM script."""

caffe_root = '/om/user/shayo/caffe/caffe/';

STIMPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
STIMPATH_NOBG = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100nobg/'

def save_features(features_perlayer, meta, layer, cnn_oi):
    features_dict = []
    features = features_perlayer[layer]
    for i,m in enumerate(meta):
        f_ = {'id':m['id'], 
            'feature':features[i,:], 
            'feature_layer':layer, 
            'model_id':cnn_oi }
        features_dict.append(f_)

    # Save output
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    with open(output_path + layer + '.pkl', 'wb') as _f:
        pk.dump(features_dict, _f)
    print 'Saved to ' + output_path + layer

def run_model(stimpath=STIMPATH, cnn_oi='VGG_S'):

    # Get image files
    meta = pk.load(open(stimpath + 'metadata.pkl'))
    filelist = [stimpath + 'images/' + m['id'] + '.png' for m in meta]
    print("There are "+str(len(filelist))+ " files")

    # Set Model Parameters
    if cnn_oi == 'caffe_reference':
        input_size = 227
        proto_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        model_file = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        model_input = '/om/user/shayo/caffe/caffe/models/bvlc_reference_caffenet'
        output_path = stimpath + 'caffe_features/'
    elif cnn_oi == 'VGG_S':
        input_size = 224
        proto_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S_deploy.prototxt'
        model_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S.caffemodel'
        model_input = '/om/user/shayo/caffe/caffe/models/VGG_CNN_S'
        output_path = stimpath + 'vgg_features/'

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
    features_perlayer = {'fc6':[], 'fc7':[], 'fc8':[]}

    for i in range(0,len(indices)-1):
        print "Processing "+str(indices[i])+" to "+ str(indices[i+1])
        images = filelist[indices[i]:indices[i+1]]
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        out = net.forward()
        features_perlayer['fc6'].append(net.blobs['fc6'].data)
        features_perlayer['fc7'].append(net.blobs['fc7'].data)
        features_perlayer['fc8'].append(net.blobs['fc8'].data)
 
    save_features(features_perlayer, meta, 'fc6', cnn_oi)
    save_features(features_perlayer, meta, 'fc7', cnn_oi)
    save_features(features_perlayer, meta, 'fc8', cnn_oi)

    return

    # Main
run_model(stimpath=STIMPATH, cnn_oi='VGG_S')
run_model(stimpath=STIMPATH, cnn_oi='caffe_reference')



