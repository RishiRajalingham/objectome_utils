import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

import random
import copy
import numpy as np
import cPickle as pk
import tabular as tb
import dldata.metrics.utils as utils
import dldata.metrics.classifier as dldat_cls
import objectome_utils as obj
import sys
import os

NPC_TRAIN = 20 
OPT_DEFAULT = {
    'classifiertype': 'svm',
    # 'classifiertype': 'softmax',
    'npc_train': NPC_TRAIN,
    'npc_test': 50,
    'n_splits': 10,
    'train_q': {},
    'test_q': {},
    'objects_oi': obj.models_combined24,
    'nsamples_noisemodel': 1,
    'noise_model': None,
    'subsample': 1000,
    'model_spec': ''
}

METRIC_KWARGS = {
    'svm': {'model_type': 'libSVM',
            'model_kwargs': {
                'C': 50000, 'class_weight': 'auto',
                'kernel': 'linear', 'probability': True}},
    'mcc': {'model_type': 'MCC2',
            'model_kwargs': {'fnorm': True, 'snorm': False}},
    'rbfsvm': {'model_type': 'libSVM',
            'model_kwargs': {'C': 50000, 'class_weight': 'auto',
            'kernel': 'rbf', 'probability': False}},
    'knn': {'model_type': 'neighbors.KNeighborsClassifier',
            'model_kwargs': {'n_neighbors': 5}},
    'softmax': {'model_type': 'LRL',
                'model_kwargs': {
                    'multi_class':'multinomial',
                    'solver': 'newton-cg'}}
}

datapath = obj.dicarlolab_homepath + 'monkey_objectome/behavioral_benchmark/data/'
feature_path = obj.dicarlolab_homepath + 'stimuli/objectome24s100/features/'
trial_path = datapath + 'trials/'
meta = obj.objectome24_meta()
uobj = list(set(meta['obj']))

img_sample_fn = datapath + 'specs/random_subsample_img_index.pkl'
imgids = pk.load(open(img_sample_fn, 'r'))
meta_id_list = list(meta['id'])
im240 = [meta_id_list.index(ii) for ii in imgids]

def get_opt(imgset, modelname, clstype='svm'):
    opt = copy.deepcopy(OPT_DEFAULT)
    opt['classifiertype'] = clstype
    opt['model_spec'] = modelname
    if imgset in ['im240', 'im240/']:
        opt['train_q'] = lambda x: (x['id'] not in set(imgids))
        opt['test_q'] = lambda x: (x['id'] in set(imgids))
        opt['n_splits'] = 10
        opt['npc_train'] = NPC_TRAIN
        opt['npc_test'] = 10
    elif imgset in ['im2400', 'im2400/']:
        opt['train_q'] = {}
        opt['test_q'] = {}
        opt['n_splits'] = 10
        opt['npc_train'] = NPC_TRAIN
        opt['npc_test'] = 100 - NPC_TRAIN
    return  opt

def normalize_features(train_features, test_features, labelset, trace_normalize=False):
    train_features, train_mean, train_std, trace = dldat_cls.normalize(
                              [train_features], trace_normalize=trace_normalize)
    train_data = {'train_mean': train_mean,
                  'train_std': train_std,
                  'trace': trace,
                  'labelset': labelset,
                  'labelmap': None}

    test_features, train_mean, train_std, trace = dldat_cls.normalize(
              [test_features], data=train_data, trace_normalize=trace_normalize)
    return test_features

def multiclass_rec_to_trials(rec, features, meta):
    trial_records = []
    for s_i, s_res in enumerate(rec['split_results']):
        train = rec['splits'][0][s_i]['train']
        test = rec['splits'][0][s_i]['test']
        train_features = features[train,:]
        test_features = features[test,:]
        
        labelset = list(s_res['labelset'])
        model = s_res['model']

        test_features = normalize_features(train_features, test_features, labelset)
        proba = model.predict_proba(test_features)

        for i,ii in enumerate(test):
            imgid = meta[ii]['id']
            true_label = meta[ii]['obj']
            true_ind = labelset.index(true_label)
            sx = proba[i,true_ind]
            for dist_label in labelset:
                if dist_label == true_label:
                    continue
                dist_ind = labelset.index(dist_label)
                sy = proba[i,dist_ind]
                prob_val = sx / (sx + sy)
                rec_curr = (true_label,) + (dist_label,) + (prob_val,) + (imgid,) + ('',) + ('',) 
                trial_records.append(rec_curr)
            
    KW_NAMES = ['sample_obj', 'dist_obj', 'prob_choice', 'id', 'WorkerID', 'AssignmentID']
    KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']
    return tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)

def multiclass_classification(features, meta, opt, outfn=None):
    evalc = {'ctypes': [('dp_standard', 'dp_standard', {'kwargs': {'error': 'std'}})],
             'labelfunc': 'obj',
             'metric_kwargs': METRIC_KWARGS[opt['classifiertype']],
             'metric_screen': 'classifier',
             'npc_test': opt['npc_test'],
             'npc_train': opt['npc_train'],
             'npc_validate': 0,
             'num_splits': opt['n_splits'],
             'split_by': 'obj',
             'split_seed': random.seed(),
             'test_q': opt['test_q'],
             'train_q': opt['train_q'],
             'use_validation': False
             }
    rec = utils.compute_metric_base(features, meta, evalc, attach_models=True, return_splits=True)
    trials = multiclass_rec_to_trials(rec, features, meta)
    if outfn is not None:
        pk.dump(trials, open(outfn, 'wb'))
    return trials

def main(argv):
    """" Input args : modelname imgset (no file/directory extensions)"""
    #print 'RUNNING: ' +argv
    modelname = argv[0]
    imgset = argv[1]
    clstype = argv[2]
    if imgset[-1] != '/':
        imgset = imgset + '/'
    opt = get_opt(imgset, modelname, clstype)

    feature_fn = feature_path + modelname + '.npy'
    if not os.path.isfile(feature_fn):
        print 'Feature not found: ' + feature_fn
        return
    features = np.load(feature_fn)
    meta = obj.objectome24_meta()
    outfn = trial_path + imgset + modelname + '_multicls' + str(NPC_TRAIN) + opt['classifiertype'] + '.pkl' 
    trials = multiclass_classification(features, meta, opt, outfn=outfn)
    print modelname, trials['prob_choice'].astype('double').mean()
    return

if __name__ == "__main__":
   main(sys.argv[1:])
   # modelname im2400 svm
