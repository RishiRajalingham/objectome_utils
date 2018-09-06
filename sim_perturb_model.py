
import numpy as np
import cPickle as pk
import scipy
import sklearn.metrics as sm
from copy import deepcopy

import h5py
import objectome_utils as obj
import objectome_model as om
import dldata.metrics.utils as utils
import random
import sys
import pandas as pd

"""  specs for task selection """
obj_oi = obj.HVM_10
SUBSAMPLED_TASKS = [[1,0],[5,0],[5,1],[8,5],[9,5], [9,8]]

# CAREFUL: meta is only for images of interest
META = obj.hvm_meta()
imgs_oi = np.array([ ((m['obj'] in obj_oi) & (m['var'] == 'V6')) for m in META])
meta = META[imgs_oi] 

imgids = []
for m in meta:
    if (m['obj'] in obj_oi) & (m['var'] == 'V6'):
        imgids.append(m['id'])
opt = {}
opt['classifiertype'] = 'softmax'
opt['model_spec'] = 'tissue_mapped'
opt['train_q'] = lambda x: (x['id'] in set(imgids))
opt['test_q'] = lambda x: (x['id'] in set(imgids))
opt['n_splits'] = 2
opt['npc_train'] = 30
opt['npc_test'] = 10

evalc = {
    'ctypes': [('dp_standard', 'dp_standard', {'kwargs': {'error': 'std'}})],
    'labelfunc': 'obj',
    'metric_kwargs': om.METRIC_KWARGS[opt['classifiertype']],
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

"""  functions for silencing experiment """

feature_path_base = '/mindhive/dicarlolab/u/rishir/models/'

def load_tissue_mapped_models(feature_fn):
    data = {}
    with h5py.File(feature_path_base + feature_fn, 'r') as f:
        data['fc6'] = f['hvm']['affine0'][:]
        data['fc7'] = f['hvm']['affine1'][:]
        data['fc7_weights'] = np.squeeze(f['hvm']['weights_affine1'][:])
        posx_fc6 = np.squeeze(f['hvm']['posx_affine0'][:])
        posy_fc6 = np.squeeze(f['hvm']['posy_affine0'][:])
    data['XY_fc6'] = np.array([ [posx_fc6[i], posy_fc6[i]] for i in range(len(posy_fc6))])
    data['dXY_fc6'] = sm.pairwise.euclidean_distances(data['XY_fc6'])
    
    return data

def load_tissue_mapped_models_old(feature_fn):
    data = {}
    with h5py.File(feature_path_base + feature_fn, 'r') as f:
        d = f['hvm_val']
        data['fc6'] = d['fc6/fc/relu'][:5760]
        data['fc7'] = d['fc7/fc/relu'][:5760]
        data['fc7_weights'] = d['fc7_weights'][:]
        posx_fc6 = d['posx_fc6'][:]
        posy_fc6 = d['posy_fc6'][:]
        posx_fc7 = d['posx_fc7'][:]
        posy_fc7 = d['posy_fc7'][:]
    data['XY_fc6'] = np.array([ [posx_fc6[i][0], posy_fc6[i][0]] for i in range(len(posy_fc6))])
    data['dXY_fc6'] = sm.pairwise.euclidean_distances(data['XY_fc7'])
    data['XY_fc7'] = np.array([ [posx_fc7[i][0], posy_fc7[i][0]] for i in range(len(posy_fc7))])
    data['dXY_fc7'] = sm.pairwise.euclidean_distances(data['XY_fc7'])
    return data

def forward_pass(fc6, fc7_w):
    fc7_r = np.dot(fc6, fc7_w)
    fc7_r = fc7_r * (fc7_r > 0)
    return fc7_r

def silence_factor_gaussian(dXY_i, sigma):
    return (1 - np.exp(-1 * (dXY_i/sigma)**2))

def silence_factor_boxcar(dXY_i, sigma):
    return np.array([1-int(d < sigma) for d in dXY_i])

def silence_factor_invert_boxcar(dXY_i, sigma):
    return np.array([1-2*int(d < sigma) for d in dXY_i])

def silence_factor_percent(dXY_i, persil):
    return np.array([1-int(d < np.percentile(dXY_i, persil)) for d in dXY_i])

def silence_features_fc6(data, delete_feat_seed=None, sigma=1.0, shuf=False, sil_type=1, n_inactivations=1):
    fc6_s = deepcopy(data['fc6'])
    fc7_w = data['fc7_weights']
    dXY = data['dXY_fc6']
    df = fc6_s.shape

    if delete_feat_seed is None:
        delete_feat_seed = random.sample(range(df[1]), n_inactivations)
        
    for dfs in delete_feat_seed:
        dXY_i = (dXY[dfs,:]) # was deepcopied
        if shuf:
            random.shuffle(dXY_i)
        
        if sil_type == 1:
            supp_factor = silence_factor_gaussian(dXY_i, sigma)
        elif sil_type == 2:
            supp_factor = silence_factor_boxcar(dXY_i, sigma)
        elif sil_type == 3:
            supp_factor = silence_factor_percent(dXY_i, sigma)
        elif sil_type == 4:
            supp_factor = silence_factor_invert_boxcar(dXY_i, sigma)
            
        supp_factor = np.tile(supp_factor, (df[0],1))
        fc6_s = np.multiply(supp_factor, fc6_s)

    fc7_s = forward_pass(fc6_s, fc7_w)
    return fc7_s, delete_feat_seed

def silence_features_fc7(data, delete_feat_seed=None, sigma=1.0, shuf=False, sil_type=1, n_inactivations=1):
    fc7_s = deepcopy(data['fc7'])
    dXY = data['dXY_fc7']
    df = fc7_s.shape

    if delete_feat_seed is None:
        delete_feat_seed = random.sample(range(df[1]), n_inactivations)
    
    
    for dfs in delete_feat_seed:
        dXY_i = (dXY[dfs,:])
        if shuf:
            random.shuffle(dXY_i)
        if sil_type == 1:
            supp_factor = silence_factor_gaussian(dXY_i, sigma)
        elif sil_type == 2:
            supp_factor = silence_factor_boxcar(dXY_i, sigma)
        elif sil_type == 3:
            supp_factor = silence_factor_percent(dXY_i, sigma)
        elif sil_type == 4:
            supp_factor = silence_factor_invert_boxcar(dXY_i, sigma)

        supp_factor = np.tile(supp_factor, (df[0],1))
        fc7_s = np.multiply(supp_factor, fc7_s)
    return fc7_s, delete_feat_seed

def get_metric_subsampled_tasks(trials, meta, niter=10, subsampled_tasks=SUBSAMPLED_TASKS, metric_baseline=None):
    if subsampled_tasks is not None:
        s,d = trials['sample_obj'], trials['dist_obj']
        inds = []
        for sti in subsampled_tasks:
            o1 = obj_oi[sti[0]]
            o2 = obj_oi[sti[1]]
            tr_in = ((s == o1) | (s == o2)) & ((d == o1) | (d == o2))
            inds.extend(np.nonzero(tr_in)[0])
        trials = trials[inds]

    out = obj.compute_behavioral_metrics(trials, meta, niter, compute_O=True, compute_I=False) 

    if metric_baseline is None:
        delta = deepcopy(out)
        delta_rel = deepcopy(out)
        delta_norm = deepcopy(out)
    else:
        delta = deepcopy(out)
        delta_rel = deepcopy(out)
        delta_norm = deepcopy(out)
        for fn in out.keys():
            if delta[fn] == []:
                continue
            for i in range(niter):
                for j in range(2):
                    delta[fn][i][j] = out[fn][i][j] - metric_baseline[fn][i][j]
                    delta_rel[fn][i][j] = (out[fn][i][j] - metric_baseline[fn][i][j]) / (out[fn][i][j] + metric_baseline[fn][i][j])
                    delta_norm[fn][i][j] = (out[fn][i][j] - metric_baseline[fn][i][j]) / (metric_baseline[fn][i][j])
    return out, delta, delta_rel, delta_norm

def get_mu_sparse(d):
    mu = np.nanmean(d)
    ntask = np.sum(np.isfinite(d)).astype('double')
    sparseness = (np.square(np.nanmean(d)) / np.nanmean(np.square(d)))
    spi = (1-sparseness) / (1-ntask**-1)
    return mu, spi     


def stats_over_exp(DATA, STATS):
    n = len(DATA)
    STATS_2D = []
    STATS_2D_fn = ['pers_i','sigma_i', 'dxy', 'rmu', 'ric1', 'ric2', 'rmu_r', 'ric1_r', 'ric2_r']
    for i in range(n):
        pers_i = STATS[i][0]
        sigma_i = STATS[i][1]
        xi,yi = STATS[i][2], STATS[i][3]
        for j in range(i+1,n):
            pers_j = STATS[j][0]
            sigma_j = STATS[j][1]

            if sigma_i == sigma_j:
                rho_out = obj.pairwise_consistency(DATA[i], DATA[j], metricn='O2_dprime_del')
                rho_out_rel = obj.pairwise_consistency(DATA[i], DATA[j], metricn='O2_dprime_rel')
                xj,yj = STATS[j][2], STATS[j][3]
                dxy = ((xi-xj)**2 + (yi-yj)**2)**0.5
                rmu = np.nanmean(rho_out['rho_n'])
                ric1 = np.nanmean(rho_out['IC_a'])
                ric2 = np.nanmean(rho_out['IC_b'])

                rmu_r = np.nanmean(rho_out_rel['rho_n'])
                ric1_r = np.nanmean(rho_out_rel['IC_a'])
                ric2_r = np.nanmean(rho_out_rel['IC_b'])
                
                STATS_2D.append([pers_i, sigma_i, dxy, rmu, ric1, ric2, rmu_r, ric1_r, ric2_r])
    return STATS_2D, STATS_2D_fn

def silencing_exp_per_sigma(data, sigma, nfeat_sample=1000, sil_layer='fc6', sil_type=1, 
    nsil_per_level=100, shuf=False, n_inactivations=1):

    features_base_all = data['fc7']
    nfeat_all = features_base_all.shape[1]
    feat_sample = np.random.choice(nfeat_all, nfeat_sample,replace=False)
    features_base = np.squeeze(features_base_all[imgs_oi,:][:,feat_sample])

    rec = utils.compute_metric_base(features_base, meta, evalc, attach_models=True, return_splits=True)
    trials_base = om.multiclass_rec_to_trials(rec, features_base, meta)
    tmp = get_metric_subsampled_tasks(trials_base, meta, subsampled_tasks=SUBSAMPLED_TASKS)
    metric_base = tmp[0]
    
    perf_baseline = obj.get_mean_behavior(metric_base, metricn='O2_dprime')
    mu_baseline, spi_baseline = get_mu_sparse(perf_baseline)
    print('Baseline perf : ' + str(mu_baseline))

    DATA, STATS_1D = [],[]
    STATS_1D_fn = [ 'perc_del', 'sigma', 'x', 'y',       
                    'mu_delta', 'mu_delta_norm', 'mu_delta_index',
                    'spi_delta', 'spi_delta_norm', 'spi_delta_index']

    for i in range(nsil_per_level):
        if sil_layer == 'fc6':
            features_s_all, delete_feat_seed = silence_features_fc6(data, delete_feat_seed=None, sigma=sigma, shuf=shuf, sil_type=sil_type, n_inactivations=n_inactivations)
        elif sil_layer == 'fc7':
            features_s_all, delete_feat_seed = silence_features_fc7(data, delete_feat_seed=None, sigma=sigma, shuf=shuf, sil_type=sil_type, n_inactivations=n_inactivations)  

        features_s = np.squeeze(features_s_all[imgs_oi,:][:,feat_sample])

        trials_s = om.multiclass_rec_to_trials_heldout_features(rec, features_base, meta, [features_s])
        metric_s, delta_s, delta_s_rel, delta_s_norm = get_metric_subsampled_tasks(trials_s[0], meta, 
            subsampled_tasks=SUBSAMPLED_TASKS, metric_baseline=metric_base)

        delta_features_norm = np.linalg.norm((features_base - features_s))
        delta_features_denom = np.linalg.norm((features_base + features_s))
        perc_del = delta_features_norm.astype('single') / delta_features_denom.astype('single')

        xy = data['XY_fc6'][delete_feat_seed[0],:]

        # tmp = {
        # 'O2_dprime':metric_s['O2_dprime'],
        # 'O3_dprime_del':delta_s['O2_dprime'],
        # 'O2_dprime_rel':delta_s_rel['O2_dprime'],
        # }
        # DATA.append(tmp)

        delta = obj.get_mean_behavior(delta_s, metricn='O2_dprime')
        delta_norm = obj.get_mean_behavior(delta_s_norm, metricn='O2_dprime')
        delta_index = obj.get_mean_behavior(delta_s_rel, metricn='O2_dprime')

        mu_delta, spi_delta = get_mu_sparse(delta)
        mu_delta_norm, spi_delta_norm = get_mu_sparse(delta_norm)
        mu_delta_index, spi_delta_index = get_mu_sparse(delta_index)
    
        STATS_1D.append([perc_del, sigma, xy[0], xy[1], 
            mu_delta, mu_delta_norm, mu_delta_index,
            spi_delta, spi_delta_norm, spi_delta_index])


    # STATS_2D, STATS_2D_fn = stats_over_exp(DATA, STATS_1D)
    #print STATS_1D
    STATS_1D = np.array(STATS_1D)
    #print 'here!'
    #print STATS_1D.shape
    S1D = pd.DataFrame(STATS_1D, columns=STATS_1D_fn)
    # # S2D = pd.DataFrame(np.array(STATS_2D), columns=STATS_2D_fn)
    # S2D = None

    return S1D


def run_model(model_fn, sil_layer='fc6', shuf_tissue_xy=False, nfeat_sample=1000, sil_type=1, n_inactivations=1):

    data = load_tissue_mapped_models(model_fn)
    ntrial_multiplier = 10.0
    STATS_1D = None
    # STATS_2D = None
    niter = 2
    for sigma in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
        if sigma < 20.0:
            nsil_per_level = 5 # rishi : switch back to 50?
        else:
            nsil_per_level = 1
        for i in range(niter):
            S1D = silencing_exp_per_sigma(data, sigma, 
                sil_layer=sil_layer, nfeat_sample=nfeat_sample, sil_type=sil_type, 
                nsil_per_level=nsil_per_level, shuf=shuf_tissue_xy, n_inactivations=n_inactivations)
            STATS_1D = pd.concat([STATS_1D, S1D])

            # DATA, S1D, S2D = silencing_exp_per_sigma(data, sigma, 
            #     sil_layer=sil_layer, nfeat_sample=nfeat_sample, sil_type=sil_type, 
            #     nsil_per_level=nsil_per_level, shuf=shuf_tissue_xy, n_inactivations=n_inactivations)
            
            
            # STATS_2D = pd.concat([STATS_2D, S2D])
            
    # out = {
    #     'STATS_1D': STATS_1D,
    #     'STATS_2D': STATS_2D,
    # }

    outfn_tup = (model_fn, sil_layer, shuf_tissue_xy, nfeat_sample, sil_type, n_inactivations)
    outfn = "tmp_data/stats1d_TM_%s_%s_shuf%d_fsample%d_sil%d_ninac%d.pkl" % outfn_tup
    STATS_1D.to_pickle(outfn)
    return

def main(argv):
    """" Input args : modelname imgset (no file/directory extensions)"""
    #print 'RUNNING: ' +argv
    print(argv)
    model_version = float(argv[0])
    shuf_tissue_xy = float(argv[1]) == 1.0
    nfeat_sample = int(argv[2])
    model_size = int(argv[3])
    sil_type = int(argv[4])
    sil_layer = 'fc6'
    n_inactivations = 1

    model_fn = 'trainval%d_4_fc6_lw10_cnn_IT%d_step_450420.h5' % (model_version, model_size)
    run_model(model_fn, sil_layer=sil_layer, shuf_tissue_xy=shuf_tissue_xy, nfeat_sample=nfeat_sample, sil_type=sil_type, n_inactivations=n_inactivations)
    return

def main_old(argv):
    """" Input args : modelname imgset (no file/directory extensions)"""
    #print 'RUNNING: ' +argv
    print(argv)
    model_version = float(argv[0])
    shuf_tissue_xy = float(argv[1]) == 1.0
    nfeat_sample = int(argv[2])
    sil_layer = argv[3]
    sil_type = int(argv[4])
    
    model_fn = 'trainval0_fc6fc7_lw%d_IT20_step_450000.h5' % model_version
    run_model(model_fn, sil_layer=sil_layer, shuf_tissue_xy=shuf_tissue_xy, nfeat_sample=nfeat_sample, sil_type=sil_type)
    return

if __name__ == "__main__":
   main(sys.argv[1:])
   # model_v shuf nfeatsample modelsize(mm) siltype
   #hard coded: sillayer,niactivations

