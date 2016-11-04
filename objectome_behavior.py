
import random
import time
import os

import pymongo
import scipy
import numpy as np
import cPickle as pk

import tabular as tb

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

KW_NAMES = ['sample_obj', 'dist_obj', 'choice', 'id', 'WorkerID', 'AssignmentID']
KW_FORMATS = ['|S40','|S40','|S40','|S40','|S40','|S40']

class psychophysDatasetObject(object):
    """ Dataset object for human psychophysics: formats data into tabarray with KW_NAMES entries. """
    def __init__(self, collection, selector, meta=None, mongo_reload=False):
        
        self.collection = collection
        self.collection_backup = BKPDATAPATH + self.collection + '_bkp.pkl'
        self.trial_kw_names = KW_NAMES
        self.trial_kw_formats = KW_FORMATS

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
        self.meta = dat['meta']
        self.splitby = dat['splitby']
        return

    def get_data(self): 

        trial_records = []
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
                    d_obj = [t_ for t_ in t_obj if t_ != s_obj][0]
                    resp = t_obj[r]
                else: #backwards compatibility with previous mturkutils
                    s_id = i[0]['id']
                    s_obj = i[0]['obj']
                    t_obj = [strip_objectomeFileName(fn) for fn in ss[1:]]
                    d_obj = [t_ for t_ in t_obj if t_ != s_obj][0]
                    resp = strip_objectomeFileName(r) 

                if  (s_id in img_oi) & (d_obj in obj_oi):
                    rec_curr = (s_obj,) + (d_obj,) + (resp,) + (s_id,) + (subj['WorkerID'],) + (subj['AssignmentID'],) 
                    trial_records.append(rec_curr)
                 
        self.trials = tb.tabarray(records=trial_records, names=self.trial_kw_names, formats=self.trial_kw_formats)
        return

def get_monkeyturk_data(dataset='objectome24'):
    if dataset == 'objectome24':
        meta_path = '/mindhive/dicarlolab/u/rishir/stimuli/objectome24s100/metadata.pkl'
        data_path = '/mindhive/dicarlolab/u/rishir/monkeyturk/allData.mat'
    
    meta = pk.load(open(meta_path,'r'))
    datmat = io.loadmat(data_path)
    uobjs = obj.models_combined24

    col_data_seg = {}
    trial_records = []
    subjs = ['Manto', 'Zico', 'Picasso', 'Nano', 'Magneto']
    for sub in subjs:
        x = datmat['allData'][sub][0,0]
        for xi in range(x.shape[0]):
            s_obj = uobjs[x[xi,0]]
            d_obj = uobjs[x[xi,2]]
            resp = uobjs[x[xi,3]]
            s_id = meta[x[xi,4]-1]['id']
            workid = sub
            assnid = 'MonkeyTurk'

            rec_curr = (s_obj,) + (d_obj,) + (resp,) + (s_id,) + (workid,) + (assnid,) 
            trial_records.append(rec_curr)

    col_data_seg['all'] = tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)
    for sub in subjs:
        t = col_data_seg['all']['WorkerID'] == sub
        col_data_seg[sub] = col_data_seg['all'][t]
    return col_data_seg

def get_model_data(dataset='objectome24'):
    if dataset == 'objectome24':
        featurespath = '/mindhive/dicarlolab/u/rishir/stimuli/objectome24s100/features/'

    meta = obj.objectome24_meta()
    all_metas, all_features = {}, {}
    # f_oi = ['ALEXNET_fc6', 'ALEXNET_fc8', 'RESNET101_conv5', 'VGG_fc6', 'VGG_fc8', 'ALEXNET_fc7', 'GOOGLENET_pool5', 'V1', 'VGG_fc7']
    f_oi = ['RESNET101_conv5']
    for f in f_oi:
        data = np.load(featurespath + f + '.npy')
        all_features[f] = data
        all_metas[f] = meta

    return obj.testFeatures(all_features, all_metas, f_oi, obj.models_combined24)    

# def get_neural_data(dataset='objectome24'):



def composite_dataset(dataset='objectome24', threshold=12000, mongo_reload=False):
    if dataset == 'objectome24':
        collections = ['objectome64', 'objectome_imglvl', 'ko_obj24_basic_2ways', 'monkobjectome']
        meta = obj.objectome24_meta()
    elif dataset == 'hvm10':
        collections = ['hvm10_basic_2ways', 'hvm10_allvar_basic_2ways'] #, 'hvm10_basic_2ways_newobj', 'hvm10-finegrain']
        meta = obj.hvm10_meta()
    fns = ['sample_obj', 'id', 'dist_obj', 'choice', 'WorkerID']
    col_data = ()
    for col in collections:
        dset = obj.psychophysDatasetObject(col, {}, meta, mongo_reload=mongo_reload)
        col_data = col_data + (dset.trials,)
    
    trials = tb.rowstack(col_data)

    #    segregate into pool and individuals
    workers = trials['WorkerID']
    col_data_seg = {'all':trials, 'pool':()}
    for uw in np.unique(workers):
        tw = np.nonzero([w == uw for w in workers])[0]
        if len(tw) < threshold:
            col_data_seg['pool'] = col_data_seg['pool'] + (trials[tw],)
        else:
            col_data_seg[uw] = trials[tw]
    col_data_seg['pool'] = tb.rowstack(col_data_seg['pool'])

    return col_data_seg

""" Behavioral metrics and utils """
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

def compute_behavioral_metrics(trials, meta, compute_metrics={'I1_dprime'}, O2_normalize=True, niter=20, rec_precomputed=None):
    rec_splithalves = {
        'O2_dprime':[],'I2_hitrate':[],'I2_dprime':[],
        'I1_hitrate':[],'I1_hitrate_c':[],'I1_hitrate_z':[],
        'I1_dprime':[],'I1_dprime_c':[],'I1_dprime_z':[]
    }

    imgdata = []
        
    if isinstance(trials, list):
        niter = len(trials)
    
    for i in range(niter):
        rec = []
        if isinstance(trials, list): # trial splits are precomputed over the raw data (e.g. reps)
            sh1,id1 = compute_behavioral_metrics_base(trials[i][0], meta, compute_metrics=compute_metrics, 
                O2_normalize=O2_normalize, rec_precomputed=rec_precomputed)
            sh2,id2 = compute_behavioral_metrics_base(trials[i][1], meta, compute_metrics=compute_metrics, 
                O2_normalize=O2_normalize, rec_precomputed=rec_precomputed)
        else: # trials are split into random halves
            ntrials = trials.shape[0]
            tr = np.arange(ntrials)
            random.shuffle(tr)
            tr1 = tr[:int(len(tr)/2)]
            tr2 = tr[int(len(tr)/2):]
            sh1,id1 = compute_behavioral_metrics_base(trials[tr1], meta, compute_metrics=compute_metrics, 
                O2_normalize=O2_normalize, rec_precomputed=rec_precomputed)
            sh2,id2 = compute_behavioral_metrics_base(trials[tr2], meta, compute_metrics=compute_metrics,
                O2_normalize=O2_normalize, rec_precomputed=rec_precomputed)

        for fnk in rec_splithalves.keys():
	    if fnk not in sh1.keys():
		continue
            rec_tmp = []
            rec_tmp.append(sh1[fnk])
            rec_tmp.append(sh2[fnk])
            rec_splithalves[fnk].append(rec_tmp)
        
        imgdata_tmp = []
        imgdata_tmp.append(id1)
        imgdata_tmp.append(id2)

        imgdata.append(imgdata_tmp)


    return rec_splithalves, imgdata

def compute_behavioral_metrics_base(trials, meta, compute_metrics={'I1_dprime'}, O2_normalize=True, rec_precomputed=None):
    uobjs = list(set(trials['sample_obj']))
    uimgs = [m['id'] for m in meta if m['obj'] in uobjs]
    imgdata = get_confusion_patterns(trials, uobjs, uimgs)
    nimgs = len(uimgs)
    nobjs = len(uobjs)

    if rec_precomputed == None:
        rec = {}
    else:
        compute_metrics = [cm for cm in compute_metrics if cm not in rec_precomputed.keys()]
        rec = rec_precomputed

    if ('O2_dprime' in compute_metrics):
        rec['O2_dprime'] = np.ones((nobjs,nobjs)) * np.nan
        for ii in range(nobjs):
            ind_i = np.nonzero(imgdata['true'][:,ii])[0]
            for jj in range(ii+1,nobjs):
                ind_j = np.nonzero(imgdata['true'][:,jj])[0]
                
                hitmat_curr = np.zeros((2,2))
                hitmat_curr[0,0] = np.sum(imgdata['choice'][ind_i,ii])
                hitmat_curr[0,1] = np.sum(imgdata['choice'][ind_i,jj])
                hitmat_curr[1,0] = np.sum(imgdata['choice'][ind_j,ii])
                hitmat_curr[1,1] = np.sum(imgdata['choice'][ind_j,jj])

                rec['O2_dprime'][ii,jj] = obj.dprime_from2x2(hitmat_curr)

    if ('I1_hitrate' in compute_metrics) | ('I2_hitrate' in compute_metrics):
        hitrates_ = imgdata['choice'] / (1.0*imgdata['selection'])
        rec['I2_hitrate'] = hitrates_
        hitrates_[imgdata['true'] > 0] = np.nan
        rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
        
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
        for fn in rec.keys():
            if 'I1' in fn:
                metric = rec[fn]
                metric_z = np.zeros(metric.shape)
                metric_c = np.zeros(metric.shape)
                for jj in range(nobjs):
                    t_i = np.nonzero(imgdata['true'][:,jj] > 0)[0]
                    metric_z[t_i] = zscore(metric[t_i])
                    metric_c[t_i] = metric[t_i] - np.mean(metric[t_i])
                rec[fn + '_z'] = metric_z
                rec[fn + '_c'] = metric_c
        
    rec['uimgs'] = uimgs
    rec['uobjs'] = uobjs


    return rec, imgdata



