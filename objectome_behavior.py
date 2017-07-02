
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
import dldata.metrics.utils as utils
import dldata.human_data.roschlib as rl

from scipy.stats import norm
from objectome import strip_objectomeFileName, get_index_MWorksRelative
from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import zscore


BKPDATAPATH = obj.dicarlolab_homepath + '/monkey_objectome/human_behaviour/data/mongodb_bkp/'

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

        nsubs = len(np.unique(self.trials['WorkerID']))
        print '%d trials from %d subjects' % (self.trials.shape[0], nsubs)
    
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
        meta_path = obj.dicarlolab_homepath + 'stimuli/objectome24s100/metadata.pkl'
        data_path = obj.dicarlolab_homepath + 'monkeyturk/allData_v2.mat'
    
    meta = pk.load(open(meta_path,'r'))
    datmat = io.loadmat(data_path)
    uobjs = obj.models_combined24

    col_data_seg = {}
    trial_records = []
    subjs = ['Manto', 'Zico', 'Picasso', 'Nano', 'Bento']#, 'Magneto', 'Pablo']
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

    col_data_seg['pool'] = tb.tabarray(records=trial_records, names=KW_NAMES, formats=KW_FORMATS)
    for sub in subjs:
        t = col_data_seg['pool']['WorkerID'] == sub
        col_data_seg[sub] = col_data_seg['pool'][t]
    return col_data_seg

def get_model_data(dataset='objectome24'):
    if dataset == 'objectome24':
        featurespath = obj.dicarlolab_homepath + 'stimuli/objectome24s100/features/'

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



def composite_dataset(dataset='objectome24', meta=None, threshold=12000, mongo_reload=False):
    if dataset == 'objectome24':
        collections = ['objectome64', 'objectome_imglvl', 'ko_obj24_basic_2ways', 'monkobjectome', 'ko_obj24_basic_2ways_mod_ver2']
        if meta == None:
            meta = obj.objectome24_meta()
    elif dataset == 'hvm10':
        collections = ['hvm10_basic_2ways', 'hvm10_allvar_basic_2ways'] #, 'hvm10_basic_2ways_newobj', 'hvm10-finegrain']
        if meta == None:
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

