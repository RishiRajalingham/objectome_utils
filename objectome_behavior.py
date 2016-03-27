
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
    
    def __init__(self, collection, selector, mongo_reload=False):
        if mongo_reload:
            conn = pymongo.Connection(port = 22334, host = 'localhost')
            db = conn.mturk
            col = db[collection]
            data = list(col.find(selector))
            print 'Loaded from mongoDB ...'

            self.collection = collection
            self.splitby = 'id'
            self.data = data
            self.meta = obj.hvm_meta()
            self.get_data()
            self.get_behavioral_metrics()

            self.backup_to_disk()
        else:
            self.collection = collection
            self.load_from_disk()
            print 'Loaded from local mongo backup...'    
    
    def get_data(self): 
        trials =  {'sample_obj':[], 'dist_obj':[], 'choice':[], 'id':[],
            'WorkerID':[], 'AssignmentID':[]}
        for subj in self.data: 
            for r,i in zip(subj['Response'], subj['ImgData']):
                s = i['Sample']
                t = i['Test']
                s_obj = s['obj']
                t_obj = [t_['obj'] for t_ in t]
                
                trials['id'].append(s['id'])
                trials['sample_obj'].append(s_obj)
                d_obj = [t_ for t_ in t_obj if t_ != s_obj]
                trials['dist_obj'].append(d_obj) 
                resp = t_obj[r]
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

    def backup_to_disk(self):
        BKFN = self.collection + '_bkp.pkl' #+ time.strftime("%m%y")
        dat = {
        'collection':self.collection,
        'data':self.data,
        'trials':self.trials,
        'rec':self.rec,
        'meta':self.meta,
        'splitby':self.splitby
        }
        with open(os.path.join(BKPDATAPATH, BKFN), 'wb') as _f:
            pk.dump(dat, _f)
        print 'Backed up to ' + BKFN

    def load_from_disk(self):
        BKFN = self.collection + '_bkp.pkl' #+ time.strftime("%m%y")
        with open(os.path.join(BKPDATAPATH, BKFN), 'r') as _f:
            dat = pk.load(_f)
        self.collection = dat['collection']
        self.data = dat['data']
        self.trials = dat['trials']
        self.rec = dat['rec']
        self.meta = dat['meta']
        self.splitby = dat['splitby']
        return

    """ Metrics and comparison utilities """

def nnan_consistency(A,B):
    ind = np.isfinite(A) & np.isfinite(B)
    return stats.spearmanr(A[ind], B[ind])[0]

def compute_behavioral_metrics(trials, meta):
    rec = {'I1':np.zeros((len(meta),1))}
    for i,m in enumerate(meta):
        t = np.nonzero(trials['id'] == m['id'])[0]
        if t.any():
            sel = trials['choice'][t]
            sam = trials['sample_obj'][t]
            rec['I1'][i] = sum(sel == sam) / len(sam)
        else:
            rec['I1'][i] = np.nan
    return rec
