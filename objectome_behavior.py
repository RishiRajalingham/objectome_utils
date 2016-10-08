
import random
import time
import os

import pymongo
import scipy
import numpy as np
import cPickle as pk

import scipy.io as io
import scipy.stats as stats
import objectome as obj
import fine_utils as fu
import dldata.metrics.utils as utils
import matplotlib.pyplot as plt
import dldata.human_data.roschlib as rl

from scipy.stats import norm
from objectome import strip_objectomeFileName, get_index_MWorksRelative
from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import zscore


BKPDATAPATH = '/mindhive/dicarlolab/u/rishir/monkey_objectome/human_behaviour/data/mongodb_bkp/'

def get_numbered_field(attribute):
    attr_idxs = {}            
    attr_index = []
    unique_attr = list(set(attribute))
    for idx, attr in enumerate(unique_attr):
        attr_idxs[attr] = idx
    for at in attribute:
        attr_index.append(attr_idxs[at])
    return unique_attr, attr_index

class psychophysDatasetObject(object):
    
    def __init__(self, collection, selector, meta=None, mongo_reload=False):
        
        self.collection = collection
        self.collection_backup = BKPDATAPATH + self.collection + '_bkp.pkl'

        if mongo_reload | (not os.path.isfile(self.collection_backup)) :
            conn = pymongo.Connection(port = 22334, host = 'localhost')
            db = conn.mturk
            col = db[collection]
            data = list(col.find(selector))
            print 'Loaded from mongoDB ...'
            self.splitby = 'id'
            self.data = data
            self.meta = meta
            self.get_data()
            self.backup_to_disk()
        else:
            self.collection = collection
            self.load_from_disk()
            print 'Loaded from local mongo backup...'    
    
    def backup_to_disk(self):
        dat = {
        'collection':self.collection,
        'data':self.data,
        'trials':self.trials,
        'meta':self.meta,
        'splitby':self.splitby
        }
        with open(self.collection_backup, 'wb') as _f:
            pk.dump(dat, _f)
        print 'Backed up to ' + self.collection_backup

    def load_from_disk(self):
        with open(self.collection_backup, 'r') as _f:
            dat = pk.load(_f)
        self.collection = dat['collection']
        self.data = dat['data']
        self.trials = dat['trials']
        trial_keys = self.trials.keys()
        for tk in trial_keys:
            self.trials[tk] = np.array(self.trials[tk])
        self.meta = dat['meta']
        self.splitby = dat['splitby']
        return

    def get_data(self): 
        trials =  {'sample_obj':[], 'dist_obj':[], 'choice':[], 'id':[],
            'WorkerID':[], 'AssignmentID':[]}
        obj_oi = np.unique(self.meta['obj'])
        img_oi = np.unique(self.meta['id'])

        for subj in self.data: 
            for r,i,ss in zip(subj['Response'], subj['ImgData'], subj['StimShown']):
                if len(i) > 1:
                    s = i['Sample']
                    t = i['Test']
                    s_id = s['id']
                    s_obj = s['obj']
                    t_obj = [t_['obj'] for t_ in t]
                    d_obj = [t_ for t_ in t_obj if t_ != s_obj]
                    resp = t_obj[r]
                else: #backwards compatibility with previous mturkutils
                    s_id = i[0]['id']
                    s_obj = i[0]['obj']
                    t_obj = [strip_objectomeFileName(fn) for fn in ss[1:]]
                    d_obj = [t_ for t_ in t_obj if t_ != s_obj]
                    resp = strip_objectomeFileName(r) 

                if  (s_id in img_oi) & (d_obj in obj_oi):
                    trials['dist_obj'].append(d_obj) 
                    trials['id'].append(s_id)
                    trials['sample_obj'].append(s_obj)
                    trials['choice'].append(resp)
                    trials['WorkerID'].append(subj['WorkerID'])
                    trials['AssignmentID'].append(subj['AssignmentID'])

        trials_keys = trials.keys()
        for fk in trials_keys:
            trials[fk] = np.array(trials[fk])
        self.trials = trials                 
        return

    def get_trial_summary(self):
        sample_obj = self.trials['sample_obj']
        dist_obj = self.trials['dist_obj']
        image_id = self.trials['id']
        resp_data = self.Response
        subj_id = self.SubjectNum
        assign_id = self.AssignmentNum
        self.trial_summary = []
        self.imgid = []
        self.choice = []

        for i in range(len(test_data)):
            trial_data = [sample_data[i], test_data[i][0], test_data[i][1], resp_data[i], subj_id[i], image_id[i], assign_id[i]]
            self.trial_summary.append(trial_data)
            self.imgid.append(image_id[i])
            if test_data[i][0] == resp_data[i]:
                self.choice.append(0)
            else:
                self.choice.append(1)
        self.trial_summary = np.array(self.trial_summary)
        self.choice = np.array(self.choice)
        return

    def subsample_trials(self, tr):
        trials_keys = self.trials.keys()
        trials_oi = {}
        tr = np.array(tr)
        for fk in trials_keys:
            trials_oi[fk] = self.trials[fk][tr]
        return trials_oi

    def get_behavioral_metrics(self):
        splitvar = self.trials[self.splitby]
        u_splitvar = list(set(splitvar))
        rec = {'I1':[[],[]]}
        niter = 10

        for itr in range(niter):
            split1, split2 = [],[]
            for i in range(len(u_splitvar)):
                tmp = np.nonzero(splitvar == u_splitvar[i])[0]
                np.random.shuffle(tmp)
                nsplit = int(len(tmp)/2)
                split1.extend(tmp[:nsplit])
                split2.extend(tmp[-nsplit:])
            
            rec['I1'][0].append( compute_behavioral_metrics(self.subsample_trials(split1), self.meta) ['I1'])
            rec['I1'][1].append( compute_behavioral_metrics(self.subsample_trials(split2), self.meta) ['I1'])
        self.rec = rec
        return 

    """ Metrics and comparison utilities """


def composite_dataset(dataset='objectome24'):
    if dataset == 'objectome24':
        collections = ['objectome64', 'objectome_imglvl', 'ko_obj24_basic_2ways', 'monkobjectome']
        meta = obj.objectome24_meta()
    fns = ['sample_obj', 'id', 'dist_obj', 'choice', 'WorkerID']
    trials = []
    for col in collections:
        dset = obj.psychophysDatasetObject(col, {}, meta, mongo_reload=False)
        trials.append(dset.trials)

    # all_trials = obj.concatenate_dictionary(trials, fns)
    
    return trials, collections

def nnan_consistency(A,B):
    ind = np.isfinite(A) & np.isfinite(B)
    return stats.spearmanr(A[ind], B[ind])[0]

def get_confusion_patterns(trials, uobjs, uimgs):
    """ returns the matrix (nimages, ntasks) of choice counts """
    nimgs = len(uimgs)
    labels = np.concatenate((uimgs, uobjs))
    
    choice = confusion_matrix(trials['id'], trials['choice'], labels)[:nimgs,nimgs:]
    true = confusion_matrix(trials['id'], trials['sample_obj'], labels)[:nimgs,nimgs:]
    selection = confusion_matrix(trials['id'], trials['dist_obj'], labels)[:len(uimgs),len(uimgs):]
    selection = selection + true

    return {'choice': np.array(choice), 
            'selection': np.array(selection), 
            'true': np.array(true)
            }

def dprime_from2x2(C):
    """ Input matrix C is essentially a 2x2 confusion matrix, 
    rows and columsn: [A| !A], but !A can be a vector"""
    maxVal = 5
    hr_ = C[0,0] / (1.0*np.nansum(C[0,:]))
    fp_ = np.nansum(C[1:,0]) / (1.0*np.nansum(C[1:,:]))
    dp = norm.ppf(hr_,0,1) - norm.ppf(fp_,0,1)
    return np.clip(dp, -maxVal, maxVal)

def compute_behavioral_metrics(trials, meta, compute_metrics={'I1_dprime'}, O2_normalize=True):
    uobjs = list(set(trials['sample_obj']))
    uimgs = [m['id'] for m in meta if m['obj'] in uobjs]
    
    imgdata = get_confusion_patterns(trials, uobjs, uimgs)
    nimgs = len(uimgs)
    nobjs = len(uobjs)

    rec = {'imgdata':imgdata}

    if ('I1_hitrate' in compute_metrics) | ('I2_hitrate' in compute_metrics):
        hitrates_ = imgdata['choice'] / (1.0*imgdata['selection'])
        hitrates_[imgdata['true'] > 0] = np.nan
        rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
        rec['I2_hitrate'] = hitrates_.flatten()

    if ('I1_dprime' in compute_metrics) | ('I2_dprime' in compute_metrics):
        hit_matrix = imgdata['choice']
        rec['I1_dprime'] = np.zeros((nimgs,1))
        rec['I2_dprime'] = np.zeros((nimgs,nobjs))

        for ii in range(nimgs):
            t = np.nonzero(imgdata['true'][ii,:] > 0)[0]
            if len(t) == 0:
                rec['I1_dprime'] [ii,:] = np.nan
                continue
            else:
                t = t[0]
            t_i = np.nonzero(imgdata['true'][:,t] == 0)[0]
            ind = np.concatenate(([ii],t_i))
            hitmat_curr = hit_matrix[ind,:]
            if ('I1_dprime' in compute_metrics):
                ind2 = list(set(range(nobjs))-set([t]))
                ind2 = np.hstack(([t], ind2))
                rec['I1_dprime'] [ii,:] = obj.dprime_from2x2(hitmat_curr[:,ind2])
            if ('I2_dprime' in compute_metrics):
                for jj in range(nobjs):
                    if imgdata['true'][ii,jj] > 0:
                        rec['I2_dprime'][ii,jj] = np.nan
                        continue
                    rec['I2_dprime'][ii,jj] = obj.dprime_from2x2(hitmat_curr[:,[t,jj]])

    if O2_normalize:
        for fn in compute_metrics:
            metric = rec[fn]
            metric_z = np.zeros(metric.shape)
            for jj in range(nobjs):
                t_i = np.nonzero(imgdata['true'][:,jj] > 0)[0]
                metric_z[t_i] = zscore(metric[t_i])
            rec[fn + '_z'] = metric_z
            
    return rec
