import scipy.io as io
import cPickle as pk
import numpy as np
import skdata.larray as larray
from dldata.stimulus_sets.hvm import ImgLoaderResizer


IMGPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
NOBGIMGPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100nobg'
HOMEPATH = '/mindhive/dicarlolab/u/rishir/monkey_objectome/machine_behaviour/'

def getPixelFeatures(objects_oi, normalize_on=False):
    """ compute pixel features on images of objects of interest """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))

    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]

    meta_ind = []
    image_paths = []
    for i, m in enumerate(meta):
        if m['obj'] in objects_oi:
            meta_ind.append(i)
            image_paths +=  [IMGPATH + 'obj64s100/' + m['id'] + '.png']

    imgs = larray.lmap(ImgLoaderResizer(inshape=(256,256), shape=(256,256), dtype='float32',normalize=normalize_on, mask=None), image_paths)
    imgs = np.array(imgs)
    ts = imgs.shape
    print ts
    pixels_features = imgs.reshape(ts[0], ts[1]*ts[2])
    pixel_meta = meta[meta_ind]
    return pixels_features, pixel_meta

def getPixelFeatures_localized(objects_oi):
    """ compute pixel features on images of objects of interest - localized to window based on metadata """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    meta_ind, image_paths, pixels_features = [], [], []
    win = 5
    img_size = 256

    for i, m in enumerate(meta):
        if m['obj'] in objects_oi:
            ii = int(-m['tz']*img_size/2 + img_size/2)
            jj = int(m['ty']*img_size/2 + img_size/2)

            meta_ind.append(i)
            fn =   [IMGPATH + 'images/' + m['obj'] + '_' + m['id'] + '.png']
            img = larray.lmap(ImgLoaderResizer(inshape=(1024,1024), shape=(img_size,img_size), dtype='float32',normalize=False, mask=None), fn)
            img = np.squeeze(np.array(img))
            
            # if image section goes beyond border, add a zero padding
            pad = np.zeros(img.shape)
            if (ii-win < 0):
                img = np.concatenate((pad, img), axis=0)
                ii += img_size
            elif (ii+win) >= img_size:
                img = np.concatenate((img, pad), axis=0)
            pad = np.zeros(img.shape)
            if (jj-win < 0):
                img = np.concatenate((pad, img), axis=1)
                jj += img_size
            elif (jj+win >= img_size):
                img = np.concatenate((img, pad), axis=1)

            tmp = img[ii-win:ii+win, jj-win:jj+win].flatten()
            pixels_features.append(tmp)
            image_paths += fn

    pixels_features = np.array(pixels_features)
    pixel_meta = meta[meta_ind]
    return pixels_features, pixel_meta

def getV1Features(objects_oi):
    """ load v1 features on images of objects of interest """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))

    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]

    meta_ind = []
    feature_paths = []
    for i, m in enumerate(meta):
        if m['obj'] in objects_oi:
            meta_ind.append(i)
            feature_paths +=  [IMGPATH + 'v1like_features/images/' + m['id'] + '.png.dat']

    v1_features = np.array([pk.load(open(fp,'r')) for fp in feature_paths])
    v1_meta = meta[meta_ind]
    return v1_features, v1_meta


def getSLFFeatures(objects_oi):
    """ SLF model features (Munch 2008) -- similar to hmax """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    SLFPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/yamins_slf/slf_trainhvm_extract_obj64s100/'
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    print 'Loading SLF features from ' + SLFPATH
    meta_ind = []
    slf_features = []
    for i, m in enumerate(meta):
        if m['obj'] in objects_oi:
            meta_ind.append(i)
            # feature_path = SLFPATH + 'slf_features/output/' + m['id'] + '.mat'
            feature_path = SLFPATH + m['id'] + '.mat'
            feature_dat = io.loadmat(feature_path)
            feature_dat = np.array(feature_dat['r'])
            slf_features.append(feature_dat)

    slf_features = np.squeeze(np.array(slf_features))
    slf_meta = meta[meta_ind]

    return slf_features, slf_meta

def getSLFFeatures_HH(objects_oi):
    """ SLF model features (Munch 2008) -- similar to hmax -- used precomputed features by hahong """
    METAPATH = '/mindhive/dicarlolab/u/rishir/lib/mturkutils/experiments/adjective_rating_fullvar/references/meta64.pkl'
    meta = pk.load(open(METAPATH, 'r'))
    f_id = pk.load(open(IMGPATH + 'hahong_features/ObjtSSFullVar3200_ids.pkl', 'r'))
    f_data = np.load(IMGPATH + 'hahong_features/ObjtSSFullVar3200_SLF.npy')
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]

    meta_ind = []
    slf_features = []
    for i, m in enumerate(meta):
        if (m['obj'] in objects_oi) & (m['id'] in f_id):
            meta_ind.append(i)
            feature_dat = f_data[f_id.index(m['id']),:]
            slf_features.append(feature_dat)

    slf_features = np.squeeze(np.array(slf_features))
    slf_meta = meta[meta_ind]

    return slf_features, slf_meta

def getNYUFeatures(objects_oi, layer=6):
    """ NYU model features (Zeiler 2013) -- see archconvnet """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    if layer == 6:
        feature_data = pk.load(open(IMGPATH + 'nyu_features/fc6.pkl', 'r'))
    elif layer == 5:
        feature_data = pk.load(open(IMGPATH + 'nyu_features/pool5.pkl', 'r'))

    fid, features = [], []
    for f in feature_data:
        fid.append(f['id'])
        features.append(f['feature'])

    features = np.array(features)
        # fid = feature_data['id']
        # features = np.array(feature_data['feature'])

    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    meta_ind = []
    for f in fid:
        ind = np.array(np.nonzero(meta['id'] == f)).flatten()[0]
        meta_ind.append(ind)

    meta_ = meta[meta_ind]
    return features, meta_

def getCaffeFeatures(objects_oi, layer=8):
    """ Caffe reference model -- from Caffe """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    
    feat_fn = 'caffe_features/fc' + str(layer) + '.pkl'
    feature_data = pk.load(open(IMGPATH + feat_fn, 'r'))
    fid, features = [], []
    for f in feature_data:
        fid.append(f['id'])
        features.append(f['feature'])

    features = np.array(features)
        
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    meta_ind = []
    for f in fid:
        ind = np.array(np.nonzero(meta['id'] == f)).flatten()[0]
        meta_ind.append(ind)

    meta_ = meta[meta_ind]
    return features, meta_

def getCaffeNOBGFeatures(objects_oi, layer=8):
    """ Caffe reference model -- from Caffe """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    
    if layer == 8:
        feature_data = np.load(IMGPATH + 'caffe_features/fc8.pkl')
    fid, features = [], []
    for f in feature_data:
        fid.append(f['id'])
        features.append(f['feature'])

    features = np.array(features)
        
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    meta_ind = []
    for f in fid:
        ind = np.array(np.nonzero(meta['id'] == f)).flatten()[0]
        meta_ind.append(ind)

    meta_ = meta[meta_ind]
    return features, meta_


def getVGGFeatures(objects_oi, layer=8):
    """ VGG features -- from Caffe """
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    
    feat_fn = 'vgg_features/fc' + str(layer) + '.pkl'
    feature_data = pk.load(open(IMGPATH + feat_fn, 'r'))
    fid, features = [], []
    for f in feature_data:
        fid.append(f['id'])
        features.append(f['feature'])

    features = np.array(features)
        
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    meta_ind = []
    for f in fid:
        ind = np.array(np.nonzero(meta['id'] == f)).flatten()[0]
        meta_ind.append(ind)

    meta_ = meta[meta_ind]
    return features, meta_

def getNeuralFeatures(objects_oi, area='IT', stim='hh'):
    meta = pk.load(open(IMGPATH + 'metadata.pkl', 'r'))
    if stim == 'hh':
        feat_fn = 'neural_features/rawdat/hh/' + str(area) + '.pkl'
    elif stim == 'ds':
        feat_fn = 'neural_features/rawdat/ds/' + str(area) + '.pkl'
    elif stim == 'all':
        f1, m1 = getNeuralFeatures(objects_oi, area=area, stim='hh')
        f2, m2 = getNeuralFeatures(objects_oi, area=area, stim='ds')
        f = np.concatenate((f1, f2), axis=len(f1.shape)-1)
        return f, m1

    feature_data = pk.load(open(IMGPATH + feat_fn, 'r'))
    fid, features = [], []
    for f in feature_data:
        fid.append(f['id'])
        features.append(f['feature'])

    features = np.array(features)
        
    """ fix obj field"""
    if len(meta[0]['obj']) == 1:
        for i,m in enumerate(meta):
            meta[i]['obj'] = m['obj'][0]
    meta_ind = []
    for f in fid:
        ind = np.array(np.nonzero(meta['id'] == f)).flatten()[0]
        meta_ind.append(ind)

    meta_ = meta[meta_ind]
    return features, meta_


def getAllFeatures(objects_oi):

    objects_oi = np.unique(objects_oi)
    all_features = {}
    all_metas = {}
    
    # all_features['PXL'], all_metas['PXL'] = getPixelFeatures(objects_oi, normalize_on=False)
    # all_features['PXLn'], all_metas['PXLn'] = getPixelFeatures(objects_oi, normalize_on=True)
    # all_features['PXL_loc'], all_metas['PXL_loc'] = getPixelFeatures_localized(objects_oi)
    # all_features['V1'], all_metas['V1'] = getV1Features(objects_oi)
    # all_features['HMAX'], all_metas['HMAX'] = getSLFFeatures_HH(objects_oi)
    # all_features['SLF'], all_metas['SLF'] = getSLFFeatures(objects_oi)
    # all_features['NYU_penult'], all_metas['NYU_penult'] = getNYUFeatures(objects_oi, layer=5)
    # all_features['NYU'], all_metas['NYU'] = getNYUFeatures(objects_oi)
    # all_features['VGG'], all_metas['VGG'] = getVGGFeatures(objects_oi, layer=8)
    # all_features['VGG_fc7'], all_metas['VGG_fc7'] = getVGGFeatures(objects_oi, layer=7)
    # all_features['VGG_fc6'], all_metas['VGG_fc6'] = getVGGFeatures(objects_oi, layer=6)
    # all_features['Caffe'], all_metas['Caffe'] = getCaffeFeatures(objects_oi, layer=8)
    # all_features['Caffe_fc7'], all_metas['Caffe_fc7'] = getCaffeFeatures(objects_oi, layer=6)
    # all_features['Caffe_fc6'], all_metas['Caffe_fc6'] = getCaffeFeatures(objects_oi, layer=6)

    # all_features['V4'], all_metas['V4'] = getNeuralFeatures(objects_oi, area='V4')
    all_features['IT'], all_metas['IT'] = getNeuralFeatures(objects_oi, area='IT', stim='all')
    # all_features['V4_rep'], all_metas['V4_rep'] = getNeuralFeatures(objects_oi, area='V4_rep')
    all_features['IT_rep'], all_metas['IT_rep'] = getNeuralFeatures(objects_oi, area='IT_rep', stim='all')

    # all_features['CaffeNOBG'], all_metas['CaffeNOBG'] = getCaffeNOBGFeatures(objects_oi)

    return all_features, all_metas

    
