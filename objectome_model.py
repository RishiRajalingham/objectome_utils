import random
import copy
import numpy as np
import cPickle as pk
import tabular as tb
import dldata.metrics.utils as utils
import objectome_utils as obj
from sklearn.decomposition import PCA
import os

OPT_DEFAULT = {
    'classifiertype':'svm',
    'npc_train':90,
    'npc_test':10,
    'n_splits':1,
    'train_q':{},
    'test_q':{},
    'objects_oi':obj.models_combined24,
    'nsamples_noisemodel':1,
    'noise_model':None,
    'subsample':1000,
    'model_spec':''
}

METRIC_KWARGS = {
    'svm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'linear', 'probability':True}},
    'mcc':  {'model_type': 'MCC2',
        'model_kwargs': {'fnorm': True, 'snorm': False}},
    'rbfsvm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'rbf', 'probability':True}},
    'knn': {'model_type': 'neighbors.KNeighborsClassifier', 
        'model_kwargs': {'n_neighbors':5}},
    'softmax': {'model_type': 'LRL', 
        'model_kwargs': {'multi_class':'multinomial'}}
}

""" ********** Feature manipulations ********** """
def sample_features(features, noise_model=None, subsample=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if noise_model == None:
        feature_sample = features
    elif noise_model == 'poisson':
        noise_mask = np.sign(features) * np.random.poisson(features)
        feature_sample = features + noise_mask
    elif noise_model == 'rep':
        fsize = features.shape
        assert len(fsize) == 3 #image x rep x site
        feature_sample = np.zeros((fsize[0],fsize[2]))
        inds = np.random.randint(0,fsize[1],fsize[0])
        for i in range(fsize[1]):
            feature_sample[inds == i,:] = features[inds == i,i,:]

    if subsample != None:
        nunits = range(feature_sample.shape[1])
        np.random.shuffle(nunits)
        feature_sample = feature_sample[:,nunits[:subsample]]
    return feature_sample

def compress_features(features, pca_thres=0.99):
    pca = PCA()
    features2 = pca.fit_transform(features)
    latent = np.cumsum(pca.explained_variance_ratio_)
    nfeat = np.nonzero(latent > pca_thres)[0][0]
    return features2[:,:nfeat]

""" ********** Classifier functions ********** """

def features2trials_pertask(features_task, meta_task, opt):
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
    rec = utils.compute_metric_base(features_task, meta_task, evalc, attach_models=False, return_splits=True)
    trial_records = []
    for s_i, s_res in enumerate(rec['split_results']):
        labelset = s_res['labelset']
        pred_label = np.array(s_res['test_prediction'])
        split_ind = rec['splits'][0][s_i]['test']
        true_label = meta_task[split_ind]['obj']
        distr_label = [np.setdiff1d(labelset,np.array(pl))[0] for pl in meta_task[split_ind]['obj']]
        assert sum(true_label == distr_label) == 0
        imgid = meta_task[split_ind]['id']
        for i in range(len(imgid)):
            rec_curr = (true_label[i],) + (distr_label[i],) + (pred_label[i],) + (imgid[i],) + ('ModelID',) + (opt['model_spec'],) 
            trial_records.append(rec_curr)
        
    KW_NAMES = ['sample_obj', 'dist_obj', 'choice', 'id', 'WorkerID', 'AssignmentID']
    KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']
    return tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)

def features2trials(features, meta, opt=OPT_DEFAULT, outfn=None):
    tasks = obj.getBinaryTasks(meta, np.array(opt['objects_oi']))
    all_trials = ()
    for isample in range(opt['nsamples_noisemodel']):
        features_sample = obj.sample_features(features, opt['noise_model'], opt['subsample'])
        for task in tasks:
            task_ind = np.squeeze(task)
            trials = features2trials_pertask(features_sample[task_ind,:], meta[task_ind], opt)
            all_trials = all_trials + (trials,)
    trials = tb.rowstack(all_trials)
    
    if outfn != None:
        pk.dump(trials, open(outfn, 'wb'))
        print 'Saved ' + outfn + ' \n ' + str(trials.shape[0])
    return trials

def features2probs_pertask_v2(features_task, meta_task, opt):
    """ instead of trials, output for each image and distracter, a probability estimate of hit rate.
        but don't use predict_proba, as this under-estimates the probability (cant be reliably estimated
        from limited data, has cross validation folds inside the trainin data). instead use decision 
        functions, renormalized to form probabilities (using a softmax). """
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
    rec = utils.compute_metric_base(features_task, meta_task, evalc, attach_models=True, return_splits=True)
    trial_records = []
    for s_i, s_res in enumerate(rec['split_results']):
        mod = s_res['model']
        split_ind = rec['splits'][0][s_i]['test']
        imgid = meta_task[split_ind]['id']
        proba = mod.predict_proba(features_task[split_ind,:])
        true_label = meta_task[split_ind]['obj']
        labelset = list(s_res['labelset'])
        distr_label = [np.setdiff1d(labelset,np.array(pl))[0] for pl in meta_task[split_ind]['obj']]

        for i, l in enumerate(true_label):
            prob_a_ = proba[i,labelset.index(l)]
            rec_curr = (true_label[i],) + (distr_label[i],) + (prob_a_,) + (imgid[i],) + ('ModelID',) + (opt['model_spec'],) 
            trial_records.append(rec_curr)
        
    KW_NAMES = ['sample_obj', 'dist_obj', 'prob_choice', 'id', 'WorkerID', 'AssignmentID']
    KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']
    return tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)


def features2probs_pertask(features_task, meta_task, opt):
    """ instead of trials, output for each image and distracter, a probability estimate of hit rate"""
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
    rec = utils.compute_metric_base(features_task, meta_task, evalc, attach_models=True, return_splits=True)
    trial_records = []
    for s_i, s_res in enumerate(rec['split_results']):
        mod = s_res['model']
        split_ind = rec['splits'][0][s_i]['test']
        imgid = meta_task[split_ind]['id']
        proba = mod.predict_proba(features_task[split_ind,:])
        true_label = meta_task[split_ind]['obj']
        labelset = list(s_res['labelset'])
        distr_label = [np.setdiff1d(labelset,np.array(pl))[0] for pl in meta_task[split_ind]['obj']]

        for i, l in enumerate(true_label):
            prob_a_ = proba[i,labelset.index(l)]
            rec_curr = (true_label[i],) + (distr_label[i],) + (prob_a_,) + (imgid[i],) + ('ModelID',) + (opt['model_spec'],) 
            trial_records.append(rec_curr)
        
    KW_NAMES = ['sample_obj', 'dist_obj', 'prob_choice', 'id', 'WorkerID', 'AssignmentID']
    KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']
    return tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)

def features2probs(features, meta, opt=OPT_DEFAULT, outfn=None):
    tasks = obj.getBinaryTasks(meta, np.array(opt['objects_oi']))
    all_trials = ()
    for isample in range(opt['nsamples_noisemodel']):
        features_sample = obj.sample_features(features, opt['noise_model'], opt['subsample'])
        for task in tasks:
            task_ind = np.squeeze(task)
            trials = features2probs_pertask(features_sample[task_ind,:], meta[task_ind], opt)
            all_trials = all_trials + (trials,)
    trials = tb.rowstack(all_trials)
    
    if outfn != None:
        pk.dump(trials, open(outfn, 'wb'))
        print 'Saved ' + outfn + ' \n ' + str(trials.shape[0])
    return trials

def run_important_ones(models=None, imgset='im240', prob_estimate=True):
    datapath = obj.dicarlolab_homepath + 'monkey_objectome/behavioral_benchmark/data/'
    feature_path = obj.dicarlolab_homepath + 'stimuli/objectome24s100/features/'
    trial_path = datapath + 'trials/'
    if models == None:
        models  = ['ALEXNET_fc6', 'ALEXNET_fc7','ALEXNET_fc8','VGG_fc6','VGG_fc7','VGG_fc8','RESNET101_conv5',
            'GOOGLENET_pool5','GOOGLENETv3_pool5','GOOGLENETv3_pool5_synth34000', 'GOOGLENETv3_pool5_retina']

    meta = obj.objectome24_meta()
    uobj = list(set(meta['obj']))

    img_sample_fn = datapath + 'specs/random_subsample_img_index.pkl'
    imgids = pk.load(open(img_sample_fn, 'r'))
    meta_id_list = list(meta['id'])
    im240 = [meta_id_list.index(ii) for ii in imgids]

    for mod in models:
        feature_fn = feature_path + mod + '.npy'
        features = np.load(feature_fn)
        if features.shape[1] > 5000:
            features = compress_features(features)
        print mod + ' : ' 
        print features.shape
        for classifiertype in ['svm', 'softmax']:#, 'mcc', 'knn']:
            opt = copy.deepcopy(OPT_DEFAULT)
            opt['model_spec'] = mod
            opt['classifiertype'] = classifiertype
            opt['subsample'] = None
            if imgset == 'im240':
                opt['train_q'] = lambda x: (x['id'] not in set(imgids))
                opt['test_q'] = lambda x: (x['id'] in set(imgids))
                opt['npc_train'] = 90
            elif imgset == 'im2400':
                opt['train_q'] = {}
                opt['test_q'] = {}
                opt['n_splits'] = 50
                opt['npc_train'] = 50
                opt['npc_test'] = 50

            outpath = trial_path + imgset + '/'
            
            if opt['subsample'] == None:
                opt['subsample'] = 0    
            if prob_estimate:
                spec_suffix = 'prob_%s.f%d.t%d' % (opt['classifiertype'],opt['subsample'],opt['npc_train'])
            else:
                spec_suffix = '%s.f%d.t%d' % (opt['classifiertype'],opt['subsample'],opt['npc_train'])
            model_spec = opt['model_spec'] + spec_suffix
            outfn = outpath + model_spec + '.pkl'
            if os.path.isfile(outfn):
                print outfn + ' exists'
                continue

            if prob_estimate:
                trials = features2probs(features, meta, opt=opt, outfn=outfn)
            else:
                trials = features2trials(features, meta, opt=opt, outfn=outfn)
    return



