import random
import itertools
import os
import scipy 
import copy
import time 
import datetime

import scipy.io as io
import cPickle as pk
import numpy as np
import skdata.larray as larray
import tabular as tb
import random

import dldata.metrics.utils as utils
import dldata.metrics.classifier as classifier
import objectome_utils as obj
from objectome_utils import get_index_MWorksRelative

HOMEPATH = '/mindhive/dicarlolab/u/rishir/monkey_objectome/machine_behaviour/'

METRIC_KWARGS = {
    'svm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'linear'}},
    'mcc':  {'model_type': 'MCC2',
        'model_kwargs': {'fnorm': True, 'snorm': False}},
    'rbfsvm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'rbf'}}
}

""" ********** Binary task formatting ********** """
def getBinaryTasks_pair(meta, object_pairs_oi):
    """ Returns binary tasks for specified pairs of objects in object_pairs_oi. """
    nc_objs = len(object_pairs_oi)
    tasks = []
    for p in range(nc_objs):
        o1 = object_pairs_oi[p][0]
        o2 = object_pairs_oi[p][1]
        trial_ind = np.nonzero((meta['obj'] == o1) | (meta['obj'] == o2))
        random.shuffle(trial_ind)
        tasks.append(np.array(trial_ind))
    return tasks

def getBinaryTasks_all(meta, objects_oi=None):
    """ Returns binary tasks for all pairs of objects in objects_oi. """
    if objects_oi == None:
        objects_oi = np.unique(meta['obj'])
        objects_oi = [i for o in objects_oi for i in o]
    nc_objs = len(objects_oi)
    pairings = list(itertools.combinations(range(nc_objs), 2))
    tasks = []
    for p in pairings:
        trial_ind = np.nonzero((meta['obj'] == objects_oi[p[0]]) | (meta['obj'] == objects_oi[p[1]]))
        random.shuffle(trial_ind)
        tasks.append(np.array(trial_ind))
    return tasks

def getBinaryTasks(meta, objects_oi):
    s = objects_oi.shape
    if len(s) == 1:
        return getBinaryTasks_all(meta, objects_oi)
    else:
        return getBinaryTasks_pair(meta, objects_oi)

""" ********** Feature manipulations ********** """
def sampleFeatures(features, noise_model=None, subsample=None):
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
        
""" ********** Classifier functions ********** """

def getClassifierRecord(features_task, meta_task, n_splits=100, classifiertype='svm'):
    if len(meta_task) == 1:
        meta_task = meta_task[0]
    features_task = np.squeeze(features_task)

    uobj = list(set(meta_task['obj']))
    train_q, test_q = {}, {}
    npc_all = [np.sum(list(meta_task['obj'] == o)) for o in uobj]
    npc = min(npc_all)
    npc_train = npc/2
    npc_test = npc/2
    metric_kwargs = METRIC_KWARGS[classifiertype]
    
    

    evalc = {'ctypes': [('dp_standard', 'dp_standard', {'kwargs': {'error': 'std'}})],
         'labelfunc': 'obj',
         'metric_kwargs': metric_kwargs,
         'metric_screen': 'classifier',
         'npc_test': npc_test,
         'npc_train': npc_train,
         'npc_validate': 0,
         'num_splits': n_splits,
         'split_by': 'obj',
         'split_seed': random.seed(),
         'test_q': test_q,
         'train_q': train_q,
         'use_validation': False}

    return utils.compute_metric_base(features_task, meta_task, evalc, attach_models=True, return_splits=True)

def runClassifierRecord(features, meta, rec, classifiertype='svm'):
    # rerun learned classifiers on (modified) features
    splits = rec['splits'][0]
    rec_test = {'split_results':[], 'splits':rec['splits']}
    metric_kwargs = copy.deepcopy(METRIC_KWARGS[classifiertype]) 

    for s_ind, split in enumerate(splits):
        model = rec['split_results'][s_ind]['model']
        test_Xy = (features[split['test'],:], meta['obj'][split['test']])
        train_data = {
            'train_mean':   rec['split_results'][s_ind]['train_mean'],
            'train_std':    rec['split_results'][s_ind]['train_std'],
            'trace':        rec['split_results'][s_ind]['trace'],
            'labelset':     rec['split_results'][s_ind]['labelset'],
            'labelmap':     rec['split_results'][s_ind]['labelmap']
            }
        model, test_result = classifier.evaluate(model,test_Xy,train_data,
            normalization=metric_kwargs.pop('normalization', True),
            trace_normalize=metric_kwargs.pop('trace_normalize', False),
            prefix='test',margins=metric_kwargs.pop('margins', False))
        rec_test['split_results'].append(test_result)
    return rec_test

def getBehavioralPatternFromRecord(rec, meta, obj_idx=None):
    nsplits = len(rec['splits'][0])
    trial_records = []

    for s_ind, split in enumerate(rec['splits']):
        labelset = rec['split_results'][s_ind]['labelset']
        split_ind = rec['splits'][0][s_ind]['test']
        pred_label = np.array(rec['split_results'][s_ind]['test_prediction'])
        true_label = meta[split_ind]['obj']
        distr_label = [np.setdiff1d(labelset,pl)[0] for pl in meta[split_ind]['obj']]
        imgid = meta[split_ind]['id']

        for i in range(len(imgid)):
            rec_curr = (true_label[i],) + (distr_label[i],) + (pred_label[i],) + (imgid[i],) + ('ModelID',) + ('ModelSpec',) 
            trial_records.append(rec_curr)
        
    KW_NAMES = ['sample_obj', 'dist_obj', 'choice', 'id', 'WorkerID', 'AssignmentID']
    KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']

    return tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)


def testFeatures_base(features, meta, task, objects_oi=None, features_s=None, nsplits=100):
    features_task = np.squeeze(features[task,:])
    meta_task = meta[task]
    if len(meta_task) == 1:
        meta_task = meta_task[0] # weird bug??

    obj_idx = {}
    for i,m in enumerate(objects_oi):
        obj_idx[m] = i
    
    rec = getClassifierRecord(features_task, meta_task, nsplits)
    return getBehavioralPatternFromRecord(rec, meta_task, obj_idx)

   
        

# def getBehavioralPatternFromRecord(rec, meta, obj_idx=None):
#     nsplits = len(rec['splits'][0])
#     trials_dict =  {'sample_obj':[], 'dist_obj':[], 'choice':[], 'id':[]}
#     trials, performance = [],[]

#     for s_ind, split in enumerate(rec['splits']):
#         labelset = rec['split_results'][s_ind]['labelset']
#         split_ind = rec['splits'][0][s_ind]['test']
#         pred_label = np.array(rec['split_results'][s_ind]['test_prediction'])
#         true_label = meta[split_ind]['obj']
#         distr_label = [np.setdiff1d(labelset,pl)[0] for pl in meta[split_ind]['obj']]
#         imgid = meta[split_ind]['id']

#         perf = (pred_label == true_label).sum() / (len(pred_label)*1.0)
#         performance.extend([perf])
        
#         trials_dict['choice'].extend(pred_label)
#         trials_dict['sample_obj'].extend(true_label)
#         trials_dict['dist_obj'].extend(distr_label)
#         trials_dict['id'].extend(imgid)

#         if obj_idx != None:
#             pred_labels_i = np.array([obj_idx[pl] for pl in pred_label])
#             actual_labels_i = np.array([obj_idx[pl] for pl in true_label])
#             distr_labels_i = np.array([obj_idx[pl] for pl in distr_label])
#             imgid_i = np.array([obj.get_index_MWorksRelative(pl) for pl in imgid])
#             trials.extend(np.array([actual_labels_i, actual_labels_i, distr_labels_i, pred_labels_i, imgid_i]).T)
#     performance = np.array(performance).mean(0)
#     trials = np.array(trials)
#     return performance, trials_dict, trials

# def testFeatures_base(features, meta, task, objects_oi=None, features_s=None, nsplits=2):
#     features_task = np.squeeze(features[task,:])
#     meta_task = meta[task]
#     if len(meta_task) == 1:
#         meta_task = meta_task[0] # weird bug??

#     obj_idx = {}
#     for i,m in enumerate(objects_oi):
#         obj_idx[m] = i
    
#     rec = getClassifierRecord(features_task, meta_task, nsplits)
#     trials = getBehavioralPatternFromRecord(rec, meta_task, obj_idx)

#     # return trials_dict
    
#     trials_s, performance_s = {}, {}
#     if features_s != None:
#         for fs in features_s:
#             if not (fs in performance_s):
#                 performance_s[fs], trials_s[fs] = [],[]
#             features_s_ = np.squeeze(features_s[fs]['features'][task,:])
#             rec_s = runClassifierRecord(features, meta, rec)
#             perf_ts_, trials_s_dict, trial_s_ = getBehavioralPatternFromRecord(rec_s, meta, obj_idx)
#             performance_s[fs].extend([perf_tmp])
#             trials_s[fs].extend(trial_s_)

#     performance = np.array(performance)
#     trials = format_trials_var(trials)
#     if features_s != None:
#         trials_s = format_trials_var(trials_s)

#     rec = {
#         'performance':performance,
#         'performance_s':performance_s,
#         'trials_dict':trials_dict,
#         'trials':trials,
#         'trials_s':trials_s
#     }
#     return rec
        
def testFeatures(all_features, all_metas, features_oi, objects_oi):
    if type(objects_oi) is dict:
        objs_oi = objects_oi['objs']
        tasks_oi = objects_oi['tasks']
    else:
        tasks_oi = np.array(objects_oi)
        objs_oi = np.array(objects_oi)

    subsample = 1000
    noise_model = None
    nsamples_noisemodel = 10
    nsplits = 100
    result = {'objs_oi':objs_oi}

    task_trials = ()
    rec = {}

    for feat in features_oi:
        if feat not in all_features.keys():
            continue
        task_trials_feat = ()
        features = all_features[feat]
        meta = fix_meta(all_metas[feat])
        tasks = getBinaryTasks(meta, tasks_oi)
        print 'Running machine_objectome : ' + str(feat) + ': ' + str(features.shape)
        for isample in range(nsamples_noisemodel):
            features_sample = sampleFeatures(features, noise_model, subsample)
            for task in tasks:
                trials = testFeatures_base(features_sample, meta, task, objs_oi, nsplits=nsplits)
                task_trials_feat = task_trials_feat + (trials,)
        task_trials = tb.rowstack(task_trials)

        task_trials['WorkerID'] = feat
        task_trials_feat = tb.rowstack(task_trials_feat)
        rec[feat] = task_trials_feat
    return task_trials

""" ********** Main functions ********** """
def computePairWiseConfusions(objects_oi, OUTPATH=None, IMGPATH=None):
    """ For a set of objects and features, run classifiers,
        and output trial structures for all 2x2 tasks.
    """
    features_oi = ['PXLn', 'V1', 'HMAX', 'Alexnet_fc6', 'Alexnet_fc7', 'Alexnet', 'VGG_fc6', 'VGG_fc7', 'VGG', 'Googlenet', 'Resnet']
    all_features, all_metas = obj.getAllFeatures(objects_oi, IMGPATH)
    result = testFeatures(all_features, all_metas, features_oi, objects_oi)
    
    for feat in features_oi:
        if feat not in all_features.keys():
            continue
        if OUTPATH != None:
            if not os.path.exists(OUTPATH):
                os.makedirs(OUTPATH)
        print 'Save to ' + OUTPATH
        save_trials(result[feat], result['objs_oi'], OUTPATH + feat + 'full_var_bg.mat')
    return 


""" New stuff? """
# def getBehavioralMetrics_base(features, meta, tasks):
#     trials =  {'sample_obj':[], 'dist_obj':[], 'choice':[], 'id':[]}
#     for task in tasks:
#         features_task = np.squeeze(features[task,:])
#         meta_task = meta[task][0]
#         rec_ = getClassifierRecord(features_task, meta_task)
#         beh_ = getBehavioralPatternFromRecord(rec_, meta_task)
#         [trials[fn].extend(beh_[fn]) for fn in trials]

#     trials_keys = trials.keys()
#     for fk in trials_keys:
#         trials[fk] = np.array(trials[fk])

#     return obj.compute_behavioral_metrics(trials, meta)

# def getBehavioralMetrics(features, meta, tasks):
#     s = features.shape
#     nim, nrep, nfeat = s[0],s[1],s[2]
#     rec = {'I1':[[],[]]}
#     niter = 10

#     for itr in range(niter):
#         trials =  {'sample_obj':[], 'dist_obj':[], 'choice':[], 'id':[]}

#         nrep_split = int(nrep/2)
#         feature_splits = {
#             'split1':np.zeros((nim, nrep_split, nfeat)),
#             'split2':np.zeros((nim, nrep_split, nfeat))}
#         for im in range(nim):
#             tmp = range(nrep)
#             np.random.shuffle(tmp)
#             feature_splits['split1'][im,:,:] = features[im,tmp[:nrep_split],:]
#             feature_splits['split2'][im,:,:] = features[im,tmp[-nrep_split:],:]
        
        
#         rec['I1'][0].append( getBehavioralMetrics_base(feature_splits['split1'].mean(1), meta, tasks)['I1'])
#         rec['I1'][1].append( getBehavioralMetrics_base(feature_splits['split2'].mean(1), meta, tasks)['I1'])

#     return rec

def run_machine_features():
    outpath = '/mindhive/dicarlolab/u/rishir/models/hvm10/' 
    meta = obj.hvm_meta()
    models_oi = np.array(obj.HVM_10)
    tasks = obj.getBinaryTasks(meta, models_oi)

    # features_n = ['fc6', 'fc7', 'fc8']
    features_n = ['fc.pkl']
    for fn in features_n:
        outn = 'caffe_' + fn + '.pkl'
        # feature_path = obj.hvm_stimpath() + 'caffe_features/' + fn + '.npy'
        feature_path = obj.hvm_stimpath() + 'alexnet++/' + fn + '.pkl'
        features = np.load(feature_path)
        rec = obj.getBehavioralMetrics(features, meta, tasks)
        with open(os.path.join(outpath, outn), 'wb') as _f:
            pk.dump(rec, _f)
    return


   
def getSplitHalfPerformance(rec, meta):
    nsplits = len(rec['splits'][0])
    perf_img = []
    for i in range(nsplits):
        split_ind = rec['splits'][0][i]['test']
        pred = rec['split_results'][i]['test_prediction']
        true = meta[split_ind]['obj']
        perf_img_ = [pred[ii] == true[ii] for ii in range(len(pred))]
        perf_img.extend(perf_img_)
    np.random.shuffle(perf_img)
    perf1 = np.mean(perf_img[-len(perf_img)/2:])
    perf2 = np.mean(perf_img[:len(perf_img)/2])

    return perf1, perf2

# def getSilencedPerformance(features_s, meta, rec, obj_idx=None):
#     nsplits = len(rec['splits'][0])
#     all_labels = np.array([obj_idx[l] for l in np.unique(meta['obj'])])
#     labelset = rec['result_summary']['labelset']
#     performance, trials = [], []
    
#     for i in range(nsplits):
#         split_ind = rec['splits'][0][i]['test']
#         model = rec['split_results'][i]['model']

#         pred_label_tmp = model.predict(features_s[split_ind,:])
#         pred_labels = np.array([labelset[pli] for pli in pred_label_tmp])
#         actual_labels = meta[split_ind]['obj']
#         hr = (pred_labels == actual_labels).sum()
#         perf = hr / (len(pred_labels)*1.0)
#         performance.extend([perf])

#         if obj_idx != None:
#             pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
#             actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
#             nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
#             tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i]).T
#             trials.extend(tmp)

#     performance = np.array(performance).mean(0)

#     return performance, trials
    
# def getTrialsFromRecord(rec, meta, obj_idx):
#     nsplits = len(rec['splits'][0])
#     all_labels = np.array([obj_idx[l] for l in np.unique(meta['obj'])])
#     labelset = rec['result_summary']['labelset']
#     trials_io, trials = [], []
#     for i in range(nsplits):
#         """ training error for ideal observer """
#         split_ind = rec['splits'][0][i]['train']
#         pred_labels = np.array(rec['split_results'][i]['train_prediction'])
#         actual_labels = meta[split_ind]['obj']
#         image_fns = meta[split_ind]['id']

#         pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
#         actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
#         nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
#         image_inds = np.array([get_index_MWorksRelative(img_i) for img_i in image_fns])
#         tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i, image_inds]).T
#         trials_io.extend(tmp)
        
#         """ testing error  """
#         split_ind = rec['splits'][0][i]['test']
#         pred_labels = np.array(rec['split_results'][i]['test_prediction'])
#         actual_labels = meta[split_ind]['obj']
#         image_fns = meta[split_ind]['id']

#         pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
#         actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
#         nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
#         image_inds = np.array([get_index_MWorksRelative(img_i) for img_i in image_fns])
#         tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i, image_inds]).T
#         trials.extend(tmp)

#     return trials, trials_io

def format_trials_var(trials):
    if (trials == None) | (trials == []):
        return trials
    elif type(trials) is dict:
        for ts in trials:
            trials[ts] = format_trials_var(trials[ts])
        return trials
    else:
        trials = np.array(trials)
        ts = trials.shape
        if len(ts) > 2:
            trials = trials.reshape(ts[0]*ts[1],ts[2])
        trials = np.array(trials).astype('double')
        return trials

# def getPerformanceFromFeatures_base(features, meta, task, objects_oi=None, features_s=None, nsplits=2):

#     features_task = np.squeeze(features[task,:])
    
#     meta_task = meta[task]
#     if len(meta_task) == 1:
#         meta_task = meta_task[0] # weird bug??

#     obj_idx = {}
#     for i,m in enumerate(objects_oi):
#         obj_idx[m] = i
    
#     rec = getClassifierRecord(features_task, meta_task, nsplits)
#     trials, trials_io = getTrialsFromRecord(rec, meta_task, obj_idx)
#     performance = 1-rec['accbal_loss']
    
#     trials_s, performance_s = {}, {}
#     if features_s != None:
#         for fs in features_s:
#             if not (fs in performance_s):
#                 performance_s[fs], trials_s[fs] = [],[]
#             features_s_ = np.squeeze(features_s[fs]['features'][task,:])
#             perf_tmp, trials_tmp = getSilencedPerformance(features_s_, meta_task, rec, obj_idx=None)
#             performance_s[fs].extend([perf_tmp])
#             trials_s[fs].extend(trials_tmp)


#     performance = np.array(performance)
#     trials = format_trials_var(trials)
#     trials_io = format_trials_var(trials_io)
#     if features_s != None:
#         trials_s = format_trials_var(trials_s)

#     return performance, performance_s, trials, trials_io, trials_s
        
def computePairWiseConfusions_base(objects_oi, OUTPATH=None, silence_mode=0):
    """ For a set of objects, compute pixel and v1 features, run classifiers,
        and output trial structures for all 2x2 tasks.
    """
    
    if type(objects_oi) is dict:
        objs_oi = objects_oi['objs']
        tasks_oi = objects_oi['tasks']
    else:
        tasks_oi = objects_oi
        objs_oi = objects_oi

    run_silencing_exp = silence_mode > 0

    print 'Loading features...'
    if run_silencing_exp:
        all_features = pk.load(open('quickload_feature_data/features.pkl', 'r'))
        all_metas = pk.load(open('quickload_feature_data/metas.pkl', 'r'))
    else:
        all_features, all_metas = obj.getAllFeatures(objs_oi)
        features_oi = ['IT', 'V4']

    result = {}
    
    for feat in features_oi:
        print 'Running machine_objectome : \n' + str(objs_oi) + '\n ' + str(feat) + '\n\n'

        features = all_features[feat]
        meta = all_metas[feat]
        tasks = getBinaryTasks(meta, tasks_oi)
        trials, trials_io = [], []
        trials_s = {}

        if run_silencing_exp:
            features_s = silenceFeature(features, silence_mode)
        else:
            features_s = None
        
        for task in tasks:
            p_, p_s_, t_, t_io_, t_s_ = getPerformanceFromFeatures_base(features, meta, task, objs_oi, features_s, nsplits=50)
            trials.extend(t_)
            trials_io.extend(t_io_)

            if run_silencing_exp:
                for fs in features_s:
                    if not (fs in t_s_):
                        trials_s[fs] = []
                    trials_s[fs].extend(t_s_[fs])

        trials = format_trials_var(trials)
        trials_io = format_trials_var(trials_io)
        if run_silencing_exp:
            trials_s = format_trials_var(trials_s)
        
        if OUTPATH != None:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%M.%H.%d%m')
            print 'Saving to ' + OUTPATH
            if not os.path.exists(OUTPATH):
                os.makedirs(OUTPATH)

            if run_silencing_exp:
                save_trials(trials, objs_oi, OUTPATH + feat + st + '.mat')
                save_trials(trials_s, objs_oi, OUTPATH + feat + st + 'sil2.mat')
            else:
                save_trials(trials, objs_oi, OUTPATH + feat + 'full_var_bg.mat')
                save_trials(trials_io, objs_oi, OUTPATH + feat + 'full_var_bg_ideal_obs.mat')

        result[feat] = trials

    return result

def computePairWiseConfusions_old(objects_oi, OUTPATH=None):
    """ For a set of objects and features, run classifiers,
        and output trial structures for all 2x2 tasks.
    """
    
    if type(objects_oi) is dict:
        objs_oi = objects_oi['objs']
        tasks_oi = objects_oi['tasks']
    else:
        tasks_oi = np.array(objects_oi)
        objs_oi = np.array(objects_oi)

    all_features, all_metas = obj.getAllFeatures(objs_oi)
    # features_oi = ['Caffe_fc6', 'Caffe_fc7', 'Caffe', 'VGG_fc6', 'VGG_fc7', 'VGG']
    features_oi = ['Alexnet_fc6', 'Alexnet_fc7', 'Alexnet']
    
    subsample = None
    nsamples_noisemodel = 2
    result = {}

    for feat in features_oi:
        if feat not in all_features.key():
            continue
        features = all_features[feat]
        meta = all_metas[feat]
        tasks = getBinaryTasks(meta, tasks_oi)
        trials, trials_io = [], []
        if 'rep' in feat:
            noise_model = 'rep' #'rep'
        else:
            noise_model = None

        print 'Running machine_objectome : ' + str(feat) + ': ' + str(features.shape) + '\t' + str(noise_model)
        for isample in range(nsamples_noisemodel):
            features_sample = sampleFeatures(features, noise_model, subsample)
            for task in tasks:
                p_, p_s_, t_, t_s_ = testFeatures_base(features, meta, task, objs_oi)
                trials.extend(t_)
	
        trials = format_trials_var(trials)
        if OUTPATH != None:
            if not os.path.exists(OUTPATH):
                os.makedirs(OUTPATH)
            if subsample != None:
                feat = feat + '_' + str(subsample)
        print 'Saved to ' + OUTPATH
        save_trials(trials, objs_oi, OUTPATH + feat + 'full_var_bg.mat')
        result[feat] = trials

    return result

def save_trials(trials, objs, outfn):
    mat_data = {}
    mat_data['data'] = trials
    mat_data['models'] = objs
    scipy.io.savemat(outfn,mat_data)
    hr = (trials[:,0] == trials[:,3])
    perf = hr.sum() / (len(hr)*1.0)
    print 'Saved (' + str(perf) + ') ' + outfn.split('/')[-1]


def quick_look(trials_s):
    import matplotlib.pyplot as plt
    HITRATE = []
    SILFRAC = []
    for fn in trials_s:
        trials_tmp = trials_s[fn]
        nT = trials_tmp.shape[1].astype('double')
        HITRATE.append( (trials_tmp[:,3] == trials_tmp[:,0]).sum() / nT )
        SILFRAC.append( float(fn[fn.find('s')+1:fn.find('r')-1]))
    HITRATE = np.array(HITRATE)
    SILFRAC = np.array(SILFRAC)

    plt.figure()
    plt.scatter(SILFRAC, HITRATE)
    plt.savefig('causality/fig/quicklook.png')
    return


def run_machine_objectome(block_num=0):
    """ Main """
    # Parse and prep
    if block_num == 0:
        IMGBLOCK = 'training8/'
        objects_oi =  np.array(obj.models_training8)
    elif block_num == 1:
        IMGBLOCK = 'testing8_block1/'
        objects_oi =  np.array(obj.models_testing8_b1)
    elif block_num == 2:
        IMGBLOCK = 'combined16/'
        objects_oi =  np.array(obj.models_combined16)
    elif block_num == 3:
        IMGBLOCK = 'testing8_block2/'
        objects_oi =  np.array(obj.models_testing8_b2)
    elif block_num == 4:
        IMGBLOCK = 'testing8_block3/'
        objects_oi =  np.array(obj.models_testing8_b3)
    elif block_num == 5:
        IMGBLOCK = 'combined24/'
        objects_oi = np.array(obj.models_combined24)
    elif block_num == 6:
        IMGBLOCK = 'obj64/'
        objects_oi = np.array(obj.models_objectome64)
    

    OUTPATH = HOMEPATH + IMGBLOCK + 'output/' 
    print 'Objectome-machine for block ' + str(block_num) + ' : ' + IMGBLOCK
    res = computePairWiseConfusions(objects_oi, OUTPATH)

def run_screens():
    obj_sets = obj.models_screentest
    for m_i, models_oi in enumerate(obj_sets):
        OUTPATH = HOMEPATH + 'screen_sets/set' + str(m_i) + '/' 
        res = computePairWiseConfusions(models_oi, OUTPATH)
    return

def run_textureless():
    models_oi = obj.models_combined24
    for i,m in enumerate(models_oi):
        models_oi[i] = m + '_tf'
    OUTPATH = HOMEPATH  + 'combined24_textureless/output/' 
    res = computePairWiseConfusions(models_oi, OUTPATH)
    return

def run_letters():
    models_oi = obj.SYMBOLS_all
    IMGPATH = '/mindhive/dicarlolab/u/rishir/stimuli/alphabet_highvar_textured/'
    OUTPATH = HOMEPATH  + 'symbols_highvar_all/output/' 
    res = computePairWiseConfusions(models_oi, OUTPATH, IMGPATH)
    return

def run_one(models_oi=None, OUTPATH=None):
    if models_oi == None:
        models_oi = obj.models_combined24
        OUTPATH = HOMEPATH + 'combined24/output/'
        # OUTPATH = HOMEPATH + 'combined24_nobg/output/'
        # OUTPATH = HOMEPATH + 'combined24_retina/output/'
        # models_oi = obj.HVM_10
        # OUTPATH = HOMEPATH + 'hvm10_retina/' + 'output/'
    res = computePairWiseConfusions(models_oi, OUTPATH)
    return

def run_causality_v1(models_oi=None, OUTPATH=None):
    if models_oi == None:
        models_oi = {'objs': np.array(obj.models_combined25), 
                    'tasks': np.array(obj.model_pairs_1)
                    }
    
    # if OUTPATH == None:   
    OUTPATH = 'causality/obj25pairs1_lashley/' 
    res = computePairWiseConfusions(models_oi, OUTPATH, 1)

    # if OUTPATH == None:
    OUTPATH = 'causality/obj25pairs1_spatialorg1/' 
    res = computePairWiseConfusions(models_oi, OUTPATH, 2)

def fix_meta(meta):
    for m in meta:
        if len(m['obj']) == 1:
            m['obj'] = m['obj'][0]
    return meta


# for blk in range(4):
#     run_machine_objectome(block_num=blk)

        # print '\n'
        # print 'Finished processing ' + feat + ' features ... '
        # print 'Total Performance : ' + str(sum(trials[:,0] == trials[:,3]) / float(trials.shape[0]))
        # for x in range(8):
        #     print objects_oi[x][-7:] + ': \t' + str(float(sum((trials[:,0] == x) & (trials[:,3] == x)))/ float(sum(trials[:,0]==x)))

        # ic = fu.binary_internal_consistency(trials)
        # print ' \n IC = ' + str(ic)

# for bn in range(4):
    # run_machine_objectome(block_num=bn)

# run_machine_objectome(block_num=6)

IMGPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'




def fix_meta_obj100s():
    """ fix objectome 100s metadata """
    import os
    import cPickle as pk
    import tabular.tab as tb
    METAFN = '/mindhive/dicarlolab/u/esolomon/objectome_32/meta64.pkl'
    IMGPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
    meta = pk.load(open(METAFN, 'r'))
    meta_new = []
    obj_filenames = os.listdir(IMGPATH + 'images')
    ofn = []
    for obj_fn in obj_filenames:
        ofn += [obj_fn.split('_')[-1].split('.')[-2]]

    meta_ind = []
    for i,m in enumerate(meta):
        if m['id'] in ofn:
            meta_ind.append(i)
    meta2 = meta[meta_ind]    

    with open(IMGPATH + 'metadata.pkl', 'wb') as _f:
        pk.dump(meta2, _f)

    return meta2





