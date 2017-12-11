import numpy as np
import cPickle as pk
import objectome_utils as obj
import glob
import sys
import os
import copy

import all_model_fns as amf
# all_models_1 = [m + '_multiclssoftmax' for m in amf.all_models]
# all_models_2 = [m + '_multiclssvm' for m in amf.all_models]
all_models = []
all_models = all_models + [m + '_multicls20softmax' for m in amf.all_models]
all_models = all_models + [m + '_multiclssoftmax' for m in amf.all_models]
all_models = all_models + [m + '_multicls20svm' for m in amf.all_models]
all_models = all_models + [m + '_multiclssvm' for m in amf.all_models]

all_subjects = amf.all_subjects


datapath = obj.dicarlolab_homepath + '/monkey_objectome/behavioral_benchmark/data/'

def get_object_metrics(fn, all_metrics):
    dfn = datapath + 'metrics/im2400/' + fn + '.pkl'
    if os.path.isfile(dfn):
        dat = pk.load(open(dfn, 'r'))
    else:
        dat = []

    metrics = {'O1_dprime', 'O1_dprime_v2', 'O2_dprime', 'I1_dprime', 'I1_dprime_v2', 'I1_dprime_C', 'I1_dprime_v2_C'}
    for m in metrics:
        mm = m
        if 'I1' in m:
            mm = m + '2400'
        if m in dat:
            all_metrics[fn][mm] = dat[m]
        else:
            all_metrics[fn][mm] = []
    return all_metrics

def get_image_metrics(fn, all_metrics):
    dfn = datapath + 'metrics/im240/' + fn + '.pkl'
    if os.path.isfile(dfn):
        dat = pk.load(open(dfn, 'r'))
    else:
        dat = []

    metrics = {'I1_dprime', 'I1_dprime_v2', 'I1_dprime_C', 'I1_dprime_v2_C', 'I2_dprime',  'I2_dprime_C', 'I2_hitrate'}
    for m in metrics:
        if m in dat:
            all_metrics[fn][m] = dat[m]
        else:
            all_metrics[fn][m] = []
    return all_metrics

def reload_metrics(fn, all_metrics):    
    all_metrics[fn] = {}
    all_metrics = get_object_metrics(fn, all_metrics)
    all_metrics = get_image_metrics(fn, all_metrics)
    return all_metrics

def get_all_metrics():
    all_metrics = {}
    all_systems = all_models + all_subjects + ['hum_pool']
    for fn in all_systems:
        all_metrics[fn] = {}
        all_metrics = get_object_metrics(fn, all_metrics)
        all_metrics = get_image_metrics(fn, all_metrics)

    all_metrics['Human Pool'] = all_metrics['hum_pool']
    all_metrics['Heldout Human Pool'] = all_metrics['hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA3Z1W0ACQDGGCA8SNALQ3K98RBAURAZWWGQBQKW']
    # all_metrics['Monkey Pool'] = all_metrics['monk_SUBPOOL_BentoMantoNanoPabloPicassoZico']
    all_metrics['Monkey Pool'] = all_metrics['monk_SUBPOOL_BentoMantoNanoPicassoZico']

    return all_metrics

def get_consistency_to_target(all_metrics, compare_models=None, target='Human Pool', metricn='I2_dprime_c', corrtype='pearson', conscorrtype='rho_n', ic_thres=0):
    stats = {}
    if compare_models is None:
        compare_models = [m for m in all_metrics.keys()]

    for model_fn in compare_models:
        if all_metrics[model_fn] == []:
            stats[model_fn] = []
        elif all_metrics[model_fn][metricn] == []:
            stats[model_fn] = []
        else:
            try:
                out = obj.pairwise_consistency(all_metrics[target], all_metrics[model_fn], metricn=metricn, corrtype=corrtype)  
                stats[model_fn] = {
                    'beh': obj.get_mean_behavior(all_metrics[model_fn], metricn),
                    'beh_sample': all_metrics[model_fn][metricn][0],
                    'cons':np.nanmean(out[conscorrtype]),
                    'cons_all':np.array(out[conscorrtype]),
                    'cons_sig':np.nanstd(out[conscorrtype]),
                    'IC_b':np.nanmean(out['IC_b']),
                    'IC_b_sig':np.nanstd(out['IC_b']),
               }
            except:
                print 'Failed to compute consistency?? ' + model_fn + ' : ' + metricn
    return stats


def compute_consistency_stats(target='Human Pool', corrtype='pearson', ignore_monk_subs=None, ignore_hum_subs=None, reload_only=None):
    metrics_outfn =  '/mindhive/dicarlolab/u/rishir/monkey_objectome/behavioral_benchmark/data/output/metrics_all.pkl'
    if reload_only is not None:
        all_metrics = pk.load(open(metrics_outfn, 'r'))
        for rl in reload_only:
            all_metrics = reload_metrics(rl, all_metrics)
            print 'Reloading '
    else:
        all_metrics = get_all_metrics()

    file_spec = '%s_%s' % (target.replace(' ', ''), corrtype)
    metrics = all_metrics[target].keys()
    
    pk.dump(all_metrics, open(metrics_outfn, 'wb'))

    hum_subpool_list_meta, monk_subpool_list_meta = amf.get_subject_list_meta(ignore_monk_subs=ignore_monk_subs, ignore_hum_subs=ignore_hum_subs)
    gnet_subpool_list_meta = amf.get_model_subject_list_meta()

    stats = {}
    for metricn in metrics:
        stats[metricn] = get_consistency_to_target(all_metrics, target=target, metricn=metricn, corrtype=corrtype)
    stats['hum_subpool_list_meta'] = hum_subpool_list_meta
    stats['monk_subpool_list_meta'] = monk_subpool_list_meta
    stats['gnet_subpool_list_meta'] = gnet_subpool_list_meta
    outfn = '/mindhive/dicarlolab/u/rishir/monkey_objectome/behavioral_benchmark/data/output/stats_' + file_spec + '_all.pkl'
    pk.dump(stats, open(outfn, 'wb'))
    print 'Done consistency stats: ', target, corrtype, outfn
    return stats

def get_residual_to_target(all_metrics, compare_models=None, target='Human Pool', metricn='I1_dprime_C2400'):
    stats = {}
    if compare_models is None:
        compare_models = [m for m in all_metrics.keys()]
    for model_fn in compare_models:
        if all_metrics[model_fn] == []:
            stats[model_fn] = []
        elif all_metrics[model_fn][metricn] == []:
            stats[model_fn] = []
        else:
            try:
                out = obj.pairwise_consistency(all_metrics[target], all_metrics[model_fn], metricn=metricn, corrtype=corrtype)  
                stats[model_fn] = {
                    'beh': obj.get_mean_behavior(all_metrics[model_fn], metricn),
                    'beh_sample': all_metrics[model_fn][metricn][0],
                    'cons':np.nanmean(out[conscorrtype]),
                    'cons_all':np.array(out[conscorrtype]),
                    'cons_sig':np.nanstd(out[conscorrtype]),
                    'IC_b':np.nanmean(out['IC_b']),
                    'IC_b_sig':np.nanstd(out['IC_b']),
               }
            except:
                print 'Failed to compute consistency?? ' + model_fn + ' : ' + metricn
    return stats

    

reload_only = None #['V1_multiclssoftmax']
compute_consistency_stats(corrtype='pearson', reload_only=reload_only)
compute_consistency_stats(corrtype='spearman', reload_only=reload_only)
