import cPickle as pk
import caffe
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import scipy.io as io
import random

""" This is a slightly edited version of Shay's OM script. - rishir"""

caffe_root = '/om/user/shayo/caffe/caffe/';
use_microsaccades = False

STIMPATH_HVM = '/mindhive/dicarlolab/u/rishir/stimuli/hvm/'
STIMPATH_HVMRET = '/mindhive/dicarlolab/u/rishir/stimuli/hvmret/'
STIMPATH_OBJ = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
STIMPATH_OBJNOBG = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100nobg/'
STIMPATH_OBJRET = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100ret/'

BATCHSIZE = 40
MAXNSTIM = 1000000000

def save_features(features_perlayer, meta, cnn_oi, output_path, repindex=None):
    layers = features_perlayer.keys()
    for layer in layers:
        features_dict = []
        features = np.array(features_perlayer[layer])
        for i,m in enumerate(meta[:MAXNSTIM]):
            f_ = {'id':m['id'], 
                'feature':features[i], 
                'feature_layer':layer, 
                'model_id':cnn_oi }
            features_dict.append(f_)

        # Save output
        if repindex == None:
            outfn = output_path + layer + '.pkl'
        else:
            outfn = output_path + 'reps/' + layer + '_' + str(repindex) + '.pkl'

        if os.path.exists(output_path) == False:
            os.mkdir(output_path)
        with open(outfn, 'wb') as _f:
            pk.dump(features_dict, _f)  
        # io.savemat(, {'features':features})
        print 'Saved to ' + outfn
    return

def get_net(stimpath, cnn_oi='VGG_S'):
    if cnn_oi == 'caffe_reference':
        input_size = 227
        proto_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        model_file = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        model_input = '/om/user/shayo/caffe/caffe/models/bvlc_reference_caffenet'
        channel_order = (2,1,0) # the reference model has channels in BGR order instead of RGB
        output_path = stimpath + 'caffe_features/'

    elif cnn_oi == 'VGG_S':
        input_size = 224
        proto_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S_deploy.prototxt'
        model_file = caffe_root + 'models/VGG_CNN_S/VGG_CNN_S.caffemodel'
        model_input = '/om/user/shayo/caffe/caffe/models/VGG_CNN_S'
        channel_order = (2,1,0) #???
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
    net.blobs['data'].reshape(BATCHSIZE,3,input_size,input_size)

    # Preprocess input data
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', channel_order)  

    return net, transformer, output_path


def translate_images(images):
    images_shifted = []
    nimages = len(images)
    for i in range(nimages):
        im = images[i]
        imsize = im.shape
        maxshift = imsize[2]/8
        shift_x = random.randint(-maxshift, maxshift)
        shift_y = random.randint(-maxshift, maxshift)
        im = np.roll(im, shift_x, axis=1)
        im = np.roll(im, shift_y, axis=2)
        images_shifted.append(im)
    return images_shifted

def run_model(stimpath, cnn_oi='VGG_S'):

    # Get image files
    meta = pk.load(open(stimpath + 'metadata.pkl'))
    filelist = [stimpath + 'images/' + m['id'] + '.png' for m in meta]
    filelist = filelist[:MAXNSTIM]
    nstim = len(filelist)
    print("There are "+str(nstim)+ " files")
    
    net, transformer, output_path = get_net(stimpath, cnn_oi)
    compute_layers = {'fc6', 'fc7', 'fc8'}
    layer_dim = {'fc6':4096, 'fc7':4096, 'fc8':1000}
    
    features_perlayer = {}
    for layer in compute_layers:
        dim = layer_dim[layer]
        features_perlayer[layer] = np.zeros([nstim, dim])

    indices = np.arange(0, nstim+1, BATCHSIZE)
    for i in range(0,len(indices)-1):
        print "Processing "+str(indices[i])+" to "+ str(indices[i+1])
        images = filelist[indices[i]:indices[i+1]]
        image_data_batch = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
        if use_microsaccades:
            net.blobs['data'].data[...] = translate_images(image_data_batch)
        else:
            net.blobs['data'].data[...] = image_data_batch
        out = net.forward()
        for layer in compute_layers:
            outdata = net.blobs[layer].data
            features_perlayer[layer][indices[i]:indices[i+1], :] = outdata
        
    return features_perlayer, meta, output_path

def format_features(stimpath, nreps=9):
    meta = pk.load(open(stimpath + 'metadata.pkl'))
    output_path = stimpath + 'caffe_features/'
    nstim = len(meta)
    compute_layers = ['fc6', 'fc7', 'fc8']
    layer_dim = {'fc6':4096, 'fc7':4096, 'fc8':1000}

    for layer in compute_layers:
        features = np.zeros((nstim,0,layer_dim[layer]))
        print layer + ' ... '
        for repindex in range(nreps):
            outfn = output_path + 'reps/' + layer + '_' + str(repindex) + '.pkl'
            with open(outfn, 'r') as _f:
                dat = pk.load(_f)
            rep_dat = np.array([imdat['feature'] for imdat in dat])
            s = rep_dat.shape
            rep_dat = rep_dat.reshape((s[0],1,s[1]))
            print '...' + str(repindex)
            features = np.concatenate((features, rep_dat), axis=1)
        print features.shape
        np.save(output_path + layer + '.npy', features)
        # with open(output_path + layer + '.pkl', 'wb') as _f:
            # pk.dump(features, _f)

    # Main

# for rep in range(9):
#     cnn_oi = 'caffe_reference'
#     features_perlayer, meta, output_path = run_model(stimpath=STIMPATH, cnn_oi=cnn_oi)
#     save_features(features_perlayer, meta, cnn_oi, output_path, rep)

#     cnn_oi = 'VGG_S'
#     features_perlayer, meta, output_path = run_model(stimpath=STIMPATH, cnn_oi=cnn_oi)
#     save_features(features_perlayer, meta, cnn_oi, output_path, rep)

def run_one(stimpath, cnn_oi):
    features_perlayer, meta, output_path = run_model(stimpath=stimpath, cnn_oi=cnn_oi)
    save_features(features_perlayer, meta, cnn_oi, output_path, repindex=None)


run_one(stimpath=STIMPATH_HVMRET, cnn_oi='caffe_reference')
run_one(stimpath=STIMPATH_HVMRET, cnn_oi='VGG_S')

