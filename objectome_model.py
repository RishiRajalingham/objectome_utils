import random
import copy
import numpy as np
import cPickle as pk
import tabular as tb
import dldata.metrics.utils as utils
import objectome_utils as obj


OPT_DEFAULT = {
    'classifiertype':'svm',
    'npc_train':50,
    'npc_test':10,
    'n_splits':10,
    'train_q':{},
    'test_q':{},
    'objects_oi':obj.models_combined24,
    'nsamples_noisemodel':10,
    'noise_model':None,
    'subsample':1000,
    'model_spec':''
}

METRIC_KWARGS = {
    'svm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'linear'}},
    'mcc':  {'model_type': 'MCC2',
        'model_kwargs': {'fnorm': True, 'snorm': False}},
    'rbfsvm': {'model_type': 'libSVM', 
        'model_kwargs': {'C': 50000, 'class_weight': 'auto',
        'kernel': 'rbf'}},
    'knn': {'model_type': 'neighbors.KNeighborsClassifier', 
        'model_kwargs': {'n_neighbors':5}}
}

        
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
    for s_ind, split in enumerate(rec['splits']):
        labelset = rec['split_results'][s_ind]['labelset']
        split_ind = rec['splits'][0][s_ind]['test']
        pred_label = np.array(rec['split_results'][s_ind]['test_prediction'])
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

def features2trials(features, meta, opt=OPT_DEFAULT):
    tasks = obj.getBinaryTasks(meta, np.array(opt['objects_oi']))
    all_trials = ()
    for isample in range(opt['nsamples_noisemodel']):
        features_sample = obj.sampleFeatures(features, opt['noise_model'], opt['subsample'])
        for task in tasks:
            task_ind = np.squeeze(task)
            trials = features2trials_pertask(features_sample[task_ind,:], meta[task_ind], opt)
            all_trials = all_trials + (trials,)
    return tb.rowstack(all_trials)


def features2trials_spec(features, meta, opt, trial_outpath):
    for subsample in [100,500,1000]:
        for npc_train in [10,30,50]:
            for classifiertype in ['svm','mcc', 'knn']:
                opt_curr = copy.deepcopy(opt)
                opt_curr['subsample'] = subsample
                opt_curr['classifiertype'] = classifiertype
                opt_curr['npc_train'] = npc_train
                spec_suffix = '%s.f%d.t%d' % (classifiertype,subsample,npc_train)
                model_spec = opt_curr['model_spec'] + spec_suffix
                opt_curr['model_spec'] = model_spec

                trials = features2trials(features, meta, opt=opt_curr)

                outfn = trial_outpath + model_spec + '.pkl'
                pk.dump(trials, open(outfn, 'wb'))
    return 



