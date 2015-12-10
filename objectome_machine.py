import random
import itertools
import sys
import os
import scipy 
import copy
import time 
import datetime

import scipy.io as io
import cPickle as pk
import numpy as np
import skdata.larray as larray

import fine_utils as fu
import dldata.metrics.utils as utils
from tabular.tab import tabarray

import objectome_utils as obj


from objectome import get_index_MWorksRelative
HOMEPATH = '/mindhive/dicarlolab/u/rishir/monkey_objectome/machine_behaviour/'

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

def getBinaryTasks_all(meta, objects_oi):
    """ Returns binary tasks for all pairs of objects in objects_oi. """
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
        
def getClassifierRecord(features_task, meta_task, n_splits=1, classifiertype='svm'):
    if len(meta_task) == 1:
        meta_task = meta_task[0]
    features_task = np.squeeze(features_task)

    uobj = list(set(meta_task['obj']))
    train_q, test_q = {}, {}
    npc_all = [np.sum(meta_task['obj'] == o) for o in uobj]
    

    npc = min(npc_all)
    npc_train = npc/2
    npc_test = npc/2
    
    metric_kwargs_svm = {'model_type': 'libSVM', 
    'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'linear'}
    }

    metric_kwargs_mcc = {'model_type': 'MCC2',
    'model_kwargs': {'fnorm': True, 'snorm': False}
    }

    if classifiertype == 'svm':
        metric_kwargs = metric_kwargs_svm
    elif classifiertype == 'mcc':
        metric_kwargs = metric_kwargs_mcc

    evalc = {'ctypes': [('dp_standard', 'dp_standard', {'kwargs': {'error': 'std'}})],
         'labelfunc': 'obj',
         'metric_kwargs': metric_kwargs,
         'metric_screen': 'classifier',
         'npc_test': npc_test,
         'npc_train': npc_train,
         'npc_validate': 0,
         'num_splits': n_splits,
         'split_by': 'obj',
         'test_q': test_q,
         'train_q': train_q,
         'use_validation': False}

    return utils.compute_metric_base(features_task, meta_task, evalc, attach_models=True, return_splits=True)
    
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


      


def getSilencedPerformance(features_s, meta, rec, obj_idx=None):
    nsplits = len(rec['splits'][0])
    labelset = rec['result_summary']['labelset']
    performance, trials = [], []
    
    for i in range(nsplits):
        split_ind = rec['splits'][0][i]['test']
        model = rec['split_results'][i]['model']

        pred_label_tmp = model.predict(features_s[split_ind,:])
        pred_labels = np.array([labelset[pli] for pli in pred_label_tmp])
        actual_labels = meta[split_ind]['obj']
        hr = (pred_labels == actual_labels).sum()
        perf = hr / (len(pred_labels)*1.0)
        performance.extend([perf])

        if obj_idx != None:
            pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
            actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
            nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
            tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i]).T
            trials.extend(tmp)

    performance = np.array(performance).mean(0)

    return performance, trials
    
def getTrialsFromRecord(rec, meta, obj_idx):
    nsplits = len(rec['splits'][0])
    all_labels = np.array([obj_idx[l] for l in np.unique(meta['obj'])])
    labelset = rec['result_summary']['labelset']
    trials_io, trials = [], []
    for i in range(nsplits):
        """ training error for ideal observer """
        split_ind = rec['splits'][0][i]['train']
        pred_labels = np.array(rec['split_results'][i]['train_prediction'])
        actual_labels = meta[split_ind]['obj']
        image_fns = meta[split_ind]['id']

        pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
        actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
        nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
        image_inds = np.array([get_index_MWorksRelative(img_i) for img_i in image_fns])
        tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i, image_inds]).T
        trials_io.extend(tmp)
        
        """ testing error  """
        split_ind = rec['splits'][0][i]['test']
        pred_labels = np.array(rec['split_results'][i]['test_prediction'])
        actual_labels = meta[split_ind]['obj']
        image_fns = meta[split_ind]['id']

        pred_labels_i = np.array([obj_idx[pl] for pl in pred_labels])
        actual_labels_i = np.array([obj_idx[pl] for pl in actual_labels])
        nonmatch_labels_i = np.array([np.setdiff1d(all_labels, pl)[0] for pl in actual_labels_i])
        image_inds = np.array([get_index_MWorksRelative(img_i) for img_i in image_fns])
        tmp = np.array([actual_labels_i, actual_labels_i, nonmatch_labels_i, pred_labels_i, image_inds]).T
        trials.extend(tmp)

    return trials, trials_io


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

def getPerformanceFromFeatures_base(features, meta, task, objects_oi=None, features_s=None, nsplits=1):

    features_task = np.squeeze(features[task,:])
    
    meta_task = meta[task]
    if len(meta_task) == 1:
        meta_task = meta_task[0] # weird bug??

    obj_idx = {}
    for i,m in enumerate(objects_oi):
        obj_idx[m] = i
    
    rec = getClassifierRecord(features_task, meta_task, nsplits)
    trials, trials_io = getTrialsFromRecord(rec, meta_task, obj_idx)
    performance = 1-rec['accbal_loss']
    
    trials_s, performance_s = {}, {}
    if features_s != None:
        for fs in features_s:
            if not (fs in performance_s):
                performance_s[fs], trials_s[fs] = [],[]
            features_s_ = np.squeeze(features_s[fs]['features'][task,:])
            perf_tmp, trials_tmp = getSilencedPerformance(features_s_, meta_task, rec, obj_idx=None)
            performance_s[fs].extend([perf_tmp])
            trials_s[fs].extend(trials_tmp)


    performance = np.array(performance)
    trials = format_trials_var(trials)
    trials_io = format_trials_var(trials_io)
    if features_s != None:
        trials_s = format_trials_var(trials_s)

    return performance, performance_s, trials, trials_io, trials_s
        

def computePairWiseConfusions(objects_oi, OUTPATH=None, silence_mode=0):
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
        # features_oi = ['VGG', 'Caffe', 'CaffeNOBG']
        features_oi = ['VGG']

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

def save_trials(trials, objs, outfn):
    mat_data = {}
    mat_data['data'] = trials
    mat_data['models'] = objs
    scipy.io.savemat(outfn,mat_data)
    hr = (trials[:,0] == trials[:,3])
    perf = hr.sum() / (len(hr)*1.0)
    print 'Saved (' + str(perf) + ') ' + outfn 


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
    print 'Objectome-machine for block ' + str(block_num) + ':' + IMGBLOCK
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

def run_one(models_oi, OUTPATH):
    OUTPATH = HOMEPATH + OUTPATH + '/output/' 
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





