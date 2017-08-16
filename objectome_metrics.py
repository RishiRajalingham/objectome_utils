import random
import copy
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from objectome_utils import nanzscore


"""
Basic metric computations from aggregated trials to random split-halves of
trials, image data (summary data structure), and metrics.
"""

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

def dprime_from2x2(C, maxVal=5):
    """ Input matrix C is essentially a 2x2 confusion matrix, 
    rows and columsn: [A| !A], but !A can be a vector""" 
    hr_ = C[0,0] / (1.0*np.nansum(C[0,:]))
    fp_ = np.nansum(C[1:,0]) / (1.0*np.nansum(C[1:,:]))
    dp = norm.ppf(hr_,0,1) - norm.ppf(fp_,0,1)
    dprime = np.clip(dp, -maxVal, maxVal)
    balacc = 0.5*(hr_ + (1-fp_))
    hitrate = hr_
    corr_rej = 1-fp_
    return dprime, balacc, hitrate, corr_rej

def trial_logicals(trials, meta):
    uobjs = list(set(meta['obj']))
    nobjs = len(uobjs)
    uimgs = list(meta['id'])
    nimgs = len(uimgs)
    use_trial_samples = 'choice' in list(trials.dtype.names)
    summary = [{'sam':[],'dist':[],'choice':[],'imgind':[]} for i in range(nobjs)]
    for i,OI in enumerate(uobjs):
        summary[i]['sam'] = (trials['sample_obj'] == OI)
        summary[i]['dist'] = (trials['dist_obj'] == OI)
        summary[i]['imgind'] = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OI) ])
        if use_trial_samples:
            summary[i]['choice'] = np.array(trials['choice'] == OI)

    return summary, use_trial_samples

def get_object_2x2(trials, meta, summary, i, use_trial_samples):
    cont_table = np.zeros((2,2))
    if use_trial_samples:
        Si_Ci = np.logical_and(summary[i]['sam'], summary[i]['choice'])
        Si_Cj = np.logical_and(summary[i]['sam'], np.logical_not(summary[i]['choice']))
        Sj_Ci = np.logical_and(summary[i]['dist'], summary[i]['choice'])
        Sj_Cj = np.logical_and(summary[i]['dist'], np.logical_not(summary[i]['choice']))
        cont_table[0,0] = np.sum(Si_Ci)
        cont_table[0,1] = np.sum(Si_Cj)
        cont_table[1,0] = np.sum(Sj_Ci)
        cont_table[1,1] = np.sum(Sj_Cj)
    else:
        xi = np.array(trials['prob_choice'][summary[i]['sam']]).astype('double')
        xj = np.array(trials['prob_choice'][summary[i]['dist']]).astype('double')
        cont_table[0,0] = np.nanmean(xi)
        cont_table[0,1] = 1-np.nanmean(xi)
        cont_table[1,0] = 1-np.nanmean(xj)
        cont_table[1,1] = np.nanmean(xj)
    return cont_table

def get_image_2x2(trials, meta, summary, img_i, use_trial_samples):
    
    uobjs = list(set(meta['obj']))
    nobjs = len(uobjs)
    uimgs = list(meta['id'])
    nimgs = len(uimgs)
    cont_table = np.zeros((2,2))
    OI = meta[meta['id'] == uimgs[img_i]]['obj']
    obj_i = uobjs.index(OI)
    tr_img_i = trials['id'] == uimgs[img_i]
    
    if use_trial_samples:

        Si_Ci = np.logical_and(tr_img_i, summary[obj_i]['choice'])
        Si_Cj = np.logical_and(tr_img_i, np.logical_not(summary[obj_i]['choice']))
        Sj_Ci = np.logical_and(summary[obj_i]['dist'], summary[obj_i]['choice'])
        Sj_Cj = np.logical_and(summary[obj_i]['dist'], np.logical_not(summary[obj_i]['choice']))
        cont_table[0,0] = np.sum(Si_Ci)
        cont_table[0,1] = np.sum(Si_Cj)
        cont_table[1,0] = np.sum(Sj_Ci)
        cont_table[1,1] = np.sum(Sj_Cj)
    else:
        xi = np.array(trials['prob_choice'][tr_img_i]).astype('double')
        xj = np.array(trials['prob_choice'][summary[obj_i]['dist']]).astype('double')
        cont_table[0,0] = np.nanmean(xi)
        cont_table[0,1] = 1-np.nanmean(xi)
        cont_table[1,0] = 1-np.nanmean(xj)
        cont_table[1,1] = np.nanmean(xj)
    return cont_table

def get_metric_augmented(trials, meta):
    """ get o1/i1 without calculating o2/i2"""
    uobjs = list(set(meta['obj']))
    nobjs = len(uobjs)
    uimgs = list(meta['id'])
    nimgs = len(uimgs)
    rec = {}
    rec['O1_dprime_v2'] = np.ones((nobjs, 1)) * np.nan
    rec['O1_dprime_v2_exp'] = np.ones((nimgs, 1)) * np.nan
    rec['I1_dprime_v2'] = np.ones((nimgs, 1)) * np.nan

    summary, use_trial_samples = trial_logicals(trials, meta)

    # loop through objects - one versus all
    for i in range(nobjs):   
        cont_table = get_object_2x2(trials, meta, summary, i, use_trial_samples)
        dp,ba,hr,cr = dprime_from2x2(cont_table)
        rec['O1_dprime_v2'][i,0] = dp
        for ii in summary[i]['imgind']:
            rec['O1_dprime_v2_exp'][ii,0] = dp

    for i in range(nimgs):
        cont_table = get_image_2x2(trials, meta, summary, i, use_trial_samples)
        dp,ba,hr,cr = dprime_from2x2(cont_table)
        rec['I1_dprime_v2'][i,0] = dp

    rec['I1_dprime_c_v2'] = rec['I1_dprime_v2'] - rec['O1_dprime_v2_exp']
   
    return rec

def get_metric_from_probs_base(trials, meta, metric_spec='all'):

    dprime_maxval = 10
    uobjs = list(set(meta['obj']))
    nobjs = len(uobjs)
    uimgs = list(meta['id'])
    nimgs = len(uimgs)

    rec = {}

    rec['O2_dprime'] = np.ones((nobjs, nobjs)) * np.nan
    rec['O2_accuracy'] = np.ones((nobjs, nobjs)) * np.nan
    rec['O2_hitrate'] = np.ones((nobjs, nobjs)) * np.nan

    rec['O2_dprime_exp'] = np.ones((nimgs, nobjs)) * np.nan
    rec['O2_accuracy_exp'] = np.ones((nimgs, nobjs)) * np.nan
    rec['O2_hitrate_exp'] = np.ones((nimgs, nobjs)) * np.nan

    rec['I2_dprime']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_accuracy']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_hitrate']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_dprime_c']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_accuracy_c']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_hitrate_c']  = np.ones((nimgs,nobjs)) * np.nan

    # precompute a bunch of logicals
    summary = [{'sam':[],'dist':[],'choice':[],'imgind':[]} for i in range(nobjs)]
    for i,OI in enumerate(uobjs):
        summary[i]['sam'] = (trials['sample_obj'] == OI)
        summary[i]['dist'] = (trials['dist_obj'] == OI)
        summary[i]['imgind'] = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OI) ])

    for i in range(nobjs):   
        img_ind_i = summary[i]['imgind']
        for j in range(i+1, nobjs):
            img_ind_j = summary[j]['imgind']

            cont_table = np.zeros((2,2))
            ti = summary[i]['sam'] & summary[j]['dist']
            tj = summary[j]['sam'] & summary[i]['dist'] 
            xi = np.array(trials['prob_choice'][ti]).astype('double')
            xj = np.array(trials['prob_choice'][tj]).astype('double')

            #xi = np.array([xx > 0.5 for xx in xi])
            #xj = np.array([xx > 0.5 for xx in xj])

            cont_table[0,0] = np.nanmean(xi)
            cont_table[0,1] = 1 - np.nanmean(xi)
            cont_table[1,0] = 1 - np.nanmean(xj)
            cont_table[1,1] = np.nanmean(xj)
            dp,ba,hr,cr = dprime_from2x2(cont_table, maxVal=dprime_maxval)
            rec['O2_dprime'][i,j] = dp
            rec['O2_accuracy'][i,j] = ba
            rec['O2_hitrate'][i,j] = hr
            rec['O2_hitrate'][j,i] = cr

            for ii,jj in zip(img_ind_i, img_ind_j):
                rec['O2_dprime_exp'][ii,j] = dp
                rec['O2_hitrate_exp'][ii,j] = hr
                rec['O2_accuracy_exp'][ii,j] = ba
                rec['O2_dprime_exp'][jj,i] = dp
                rec['O2_hitrate_exp'][jj,i] = cr
                rec['O2_accuracy_exp'][jj,i] = ba

        if metric_spec != 'all':
            continue

        for j in range(nimgs):
            t_j = trials['id'] == uimgs[j] 
            OJ = meta[meta['id'] == uimgs[j]]['obj']
            j_ = uobjs.index(OJ)
            if j_ == i:
                continue
            cont_table = np.zeros((2,2))
            # read this next line as trials with: sample j, distracter I, choice J.

            ti = t_j & summary[i]['dist'] 
            tj = summary[i]['sam'] & summary[j_]['dist']
            xi = np.array(trials['prob_choice'][ti]).astype('double')
            xj = np.array(trials['prob_choice'][tj]).astype('double')

            cont_table[0,0] = np.nanmean(xi)
            cont_table[0,1] = 1 - np.nanmean(xi)
            cont_table[1,0] = 1 - np.nanmean(xj)
            cont_table[1,1] = np.nanmean(xj)

            dp,ba,hr,cr = dprime_from2x2(cont_table, maxVal=dprime_maxval)
            rec['I2_dprime'][j,i] = dp
            rec['I2_accuracy'][j,i] = ba
            rec['I2_hitrate'][j,i] = hr

    rec['I2_dprime_c']= rec['I2_dprime'] - rec['O2_dprime_exp']
    rec['I2_accuracy_c'] = rec['I2_accuracy'] - rec['O2_accuracy_exp']
    rec['I2_hitrate_c'] = rec['I2_hitrate'] - rec['O2_hitrate_exp']

    rec['I1_dprime'] = np.nanmean(rec['I2_dprime'], 1)
    rec['I1_accuracy'] = np.nanmean(rec['I2_accuracy'], 1)
    rec['I1_hitrate'] = np.nanmean(rec['I2_hitrate'], 1)

    rec['I1_dprime_c'] = np.nanmean(rec['I2_dprime_c'], 1)
    rec['I1_accuracy_c'] = np.nanmean(rec['I2_accuracy_c'], 1)
    rec['I1_hitrate_c'] = np.nanmean(rec['I2_hitrate_c'], 1)

    return rec

def get_metric_from_trials_base(trials, meta, metric_spec='all'):
          
    uobjs = list(set(meta['obj']))
    nobjs = len(uobjs)
    uimgs  = list(meta['id'])
    nimgs = len(uimgs)
         
    rec = {}

    rec['O2_dprime'] = np.ones((nobjs, nobjs)) * np.nan
    rec['O2_accuracy'] = np.ones((nobjs, nobjs)) * np.nan
    rec['O2_hitrate'] = np.ones((nobjs, nobjs)) * np.nan

    rec['O2_dprime_exp'] = np.ones((nimgs, nobjs)) * np.nan
    rec['O2_accuracy_exp'] = np.ones((nimgs, nobjs)) * np.nan
    rec['O2_hitrate_exp'] = np.ones((nimgs, nobjs)) * np.nan

    rec['I2_dprime']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_accuracy']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_hitrate']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_dprime_c']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_accuracy_c']  = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_hitrate_c']  = np.ones((nimgs,nobjs)) * np.nan

    # precompute a bunch of logicals
    summary = [{'sam':[],'dist':[],'choice':[],'imgind':[]} for i in range(nobjs)]
    for i,OI in enumerate(uobjs):
        summary[i]['sam'] = np.array(trials['sample_obj'] == OI)
        summary[i]['dist'] = np.array(trials['dist_obj'] == OI)
        summary[i]['choice'] = np.array(trials['choice'] == OI)
        summary[i]['imgind'] = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OI) ])

    for i in range(nobjs):   
        img_ind_i = summary[i]['imgind']
        for j in range(i+1, nobjs):
            img_ind_j = summary[j]['imgind']

            Si_Dj = np.logical_and(summary[i]['sam'], summary[j]['dist'])
            Si_Dj_i = np.logical_and(Si_Dj, summary[i]['choice'])
            Si_Dj_j = np.logical_and(Si_Dj, summary[j]['choice'])

            Sj_Di = np.logical_and(summary[j]['sam'], summary[i]['dist'])
            Sj_Di_i = np.logical_and(Sj_Di, summary[i]['choice'])
            Sj_Di_j = np.logical_and(Sj_Di, summary[j]['choice'])

            cont_table = np.zeros((2,2))
            cont_table[0,0] = np.sum(Si_Dj_i)
            cont_table[0,1] = np.sum(Si_Dj_j)
            cont_table[1,0] = np.sum(Sj_Di_i)
            cont_table[1,1] = np.sum(Sj_Di_j)
            dp,ba,hr,cr = dprime_from2x2(cont_table)
            rec['O2_dprime'][i,j] = dp
            rec['O2_accuracy'][i,j] = ba
            rec['O2_hitrate'][i,j] = hr
            rec['O2_hitrate'][j,i] = cr

            for ii,jj in zip(img_ind_i, img_ind_j):
                rec['O2_dprime_exp'][ii,j] = dp
                rec['O2_hitrate_exp'][ii,j] = hr
                rec['O2_accuracy_exp'][ii,j] = ba
                rec['O2_dprime_exp'][jj,i] = dp
                rec['O2_hitrate_exp'][jj,i] = cr
                rec['O2_accuracy_exp'][jj,i] = ba

        if metric_spec != 'all':
            continue

        for j in range(nimgs):
            t_j = np.array(trials['id'] == uimgs[j])
            OJ = meta[meta['id'] == uimgs[j]]['obj']
            j_ = uobjs.index(OJ)
            if j_ == i:
                continue
            cont_table = np.zeros((2,2))

            Sj_Di = np.logical_and(t_j, summary[i]['dist'])
            Sj_Di_j = np.logical_and(Sj_Di, summary[j_]['choice'])
            Sj_Di_i = np.logical_and(Sj_Di, summary[i]['choice'])

            Si_Dj = np.logical_and(summary[i]['sam'], summary[j_]['dist'])
            Si_Dj_j = np.logical_and(Si_Dj, summary[j_]['choice'])
            Si_Dj_i = np.logical_and(Si_Dj, summary[i]['choice'])

            # read this next line as trials with: sample j, distracter I, choice J.
            cont_table[0,0] = np.sum(Sj_Di_j)
            cont_table[0,1] = np.sum(Sj_Di_i)
            cont_table[1,0] = np.sum(Si_Dj_j)
            cont_table[1,1] = np.sum(Si_Dj_i)

            dp,ba,hr,cr = dprime_from2x2(cont_table)
            rec['I2_dprime'][j,i] = dp
            rec['I2_accuracy'][j,i] = ba
            rec['I2_hitrate'][j,i] = hr

    rec['I2_dprime_c']= rec['I2_dprime'] - rec['O2_dprime_exp']
    rec['I2_accuracy_c'] = rec['I2_accuracy'] - rec['O2_accuracy_exp']
    rec['I2_hitrate_c'] = rec['I2_hitrate'] - rec['O2_hitrate_exp']

    rec['I1_dprime'] = np.nanmean(rec['I2_dprime'], 1)
    rec['I1_accuracy'] = np.nanmean(rec['I2_accuracy'], 1)
    rec['I1_hitrate'] = np.nanmean(rec['I2_hitrate'], 1)

    rec['I1_dprime_c'] = np.nanmean(rec['I2_dprime_c'], 1)
    rec['I1_accuracy_c'] = np.nanmean(rec['I2_accuracy_c'], 1)
    rec['I1_hitrate_c'] = np.nanmean(rec['I2_hitrate_c'], 1)

    return rec

def get_metric_base(trials, meta, metric_spec='all'):
    if 'choice' in list(trials.dtype.names):
        return get_metric_from_trials_base(trials, meta, metric_spec=metric_spec)
    else:
        return get_metric_from_probs_base(trials, meta, metric_spec=metric_spec)

def update_metric(metric_dat_, meta):

    uobj = list(set(meta['obj']))
    uimg = list(meta['id'])

    # I1 normalize
    DAT = metric_dat_['I1_dprime']
    OUT_C = copy.deepcopy(DAT)
    OUT_Z = copy.deepcopy(DAT)

    for uo in uobj:
        tid = meta[meta['obj'] == uo]['id']
        ind = [uimg.index(ti) for ti in tid]
        for i in range(len(DAT)):
            for j in range(2):
                tmp = DAT[i][j][ind]
                mu, sig = np.nanmean(tmp), np.nanstd(tmp)
                for ii in ind:
                    OUT_C[i][j][ii] = (DAT[i][j][ii] - mu)
                    OUT_Z[i][j][ii] = (DAT[i][j][ii] - mu)/sig
    metric_dat_['I1_dprime_C'] = OUT_C
    metric_dat_['I1_dprime_Z'] = OUT_Z

    # I2 normalize
    DAT = metric_dat_['I2_dprime']
    OUT_C = copy.deepcopy(DAT)
    OUT_Z = copy.deepcopy(DAT)

    for oi,uo in enumerate(uobj):
        tid = meta[meta['obj'] == uo]['id']
        ind = [uimg.index(ti) for ti in tid]
        for oj,uo2 in enumerate(uobj):
            for i in range(len(DAT)):
                for j in range(2):
                    tmp = DAT[i][j][ind,oj]
                    mu, sig = np.nanmean(tmp), np.nanstd(tmp)
                    for ii in ind:
                        OUT_C[i][j][ii,oj] = (DAT[i][j][ii,oj] - mu)
                        OUT_Z[i][j][ii,oj] = (DAT[i][j][ii,oj] - mu)/sig
    metric_dat_['I2_dprime_C'] = OUT_C
    metric_dat_['I2_dprime_Z'] = OUT_Z
    return metric_dat_

        
def compute_behavioral_metrics(trials, meta, niter, metric_spec='all', noise_model='trial_samples'):
    metrics = [
        'O2_dprime', 'O2_accuracy', 'O2_hitrate', 
        'I2_dprime', 'I2_accuracy', 'I2_hitrate', 
        'I1_dprime', 'I1_accuracy', 'I1_hitrate', 
        'I2_dprime_c', 'I2_accuracy_c', 'I2_hitrate_c', 
        'I1_dprime_c', 'I1_accuracy_c', 'I1_hitrate_c'
        ]

    metrics_2 = ['I1_dprime_v2', 'O1_dprime_v2', 'I1_dprime_c_v2']

    if noise_model == None:
        # run on all trials just once without any sampling
        rec_all = get_metric_base(trials, meta, metric_spec=metric_spec)
        rec_a_all = get_metric_augmented(trials, meta)
    rec = {k: [] for k in metrics + metrics_2}
    
    ntrials = trials.shape[0]
    for i in range(niter):
        if noise_model == None:
            rec1 = rec_all
            rec2 = rec_all
            rec_a1 = rec_a_all
            rec_a2 = rec_a_all
        else: # if noise_model == 'trial_samples':
            tr = np.arange(ntrials)
            random.shuffle(tr)
            tr1 = tr[:int(len(tr)/2)]
            tr2 = tr[int(len(tr)/2):]
            rec1 = get_metric_base(trials[tr1], meta, metric_spec=metric_spec)
            rec2 = get_metric_base(trials[tr2], meta, metric_spec=metric_spec)
            rec_a1 = get_metric_augmented(trials[tr1], meta)
            rec_a2 = get_metric_augmented(trials[tr2], meta)

        for fn in metrics:
            rec[fn].append([rec1[fn], rec2[fn]])
        for fn in metrics_2:
            rec[fn].append([rec_a1[fn], rec_a2[fn]])

    rec = update_metric(rec, meta)
    return rec
    
def augment_behavioral_metrics(trials, meta, rec_precomputed, noise_model='trial_samples'):
    if noise_model == None:
        # run on all trials just once without any sampling
        rec_all = get_metric_augmented(trials, meta)
    metrics = ['I1_dprime_v2', 'O1_dprime_v2', 'I1_dprime_c_v2']
    rec = rec_precomputed
    for k in metrics:
        rec[k] = []
    
    niter = len(rec['O2_dprime'])
    niter = min(niter,2)
    
    ntrials = trials.shape[0]
    for i in range(niter):
        if noise_model == None:
            rec1 = rec_all
            rec2 = rec_all
        else: # if noise_model == 'trial_samples':
            tr = np.arange(ntrials)
            random.shuffle(tr)
            tr1 = tr[:int(len(tr)/2)]
            tr2 = tr[int(len(tr)/2):]
            rec1 = get_metric_augmented(trials[tr1], meta)
            rec2 = get_metric_augmented(trials[tr2], meta)
        for fn in metrics:
            rec[fn].append([rec1[fn], rec2[fn]])
    return rec

"""
Methods for measuring consistency of output metric
"""
def get_mean_behavior(b1, metricn):
    return np.squeeze(np.nanmean(np.nanmean(b1[metricn], axis=1), axis=0))

def nnan_consistency(A,B, corrtype='pearson', ignore_vals=None):
    ind = np.isfinite(A) & np.isfinite(B)
    A,B = A[ind], B[ind]
    if ignore_vals is not None:
        A2,B2 = [],[]
        for a,b in zip(A,B):
            if not((a in ignore_vals) or (b in ignore_vals)):
                A2.append(a)
                B2.append(b)
        A = np.array(A2)
        B = np.array(B2)
    if corrtype == 'pearson':
        return pearsonr(A, B)[0]
    elif corrtype == 'spearman':
        return spearmanr(A, B)[0]

def pairwise_consistency(A,B, metricn='I1_dprime_z', corrtype='pearson', img_subsample=None, ignore_vals=None):
    niter = min(len(A[metricn]), len(B[metricn]))
    out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[], 'rho_n_sq':[]}
    for i in range(niter):
        a0,a1 = A[metricn][i][0], A[metricn][i][1]
        b0,b1 = B[metricn][i][0], B[metricn][i][1]
        
        if img_subsample is not None:
            if a0.ndim == 1:
                a0, a1 = a0[img_subsample], a1[img_subsample]
                b0, b1 = b0[img_subsample], b1[img_subsample]
            else:
                a0, a1 = a0[img_subsample, :], a1[img_subsample, :]
                b0, b1 = b0[img_subsample, :], b1[img_subsample, :]
        ind = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(b0) & np.isfinite(b1)

        a0 = np.squeeze(a0[ind])
        a1 = np.squeeze(a1[ind])
        b0 = np.squeeze(b0[ind])
        b1 = np.squeeze(b1[ind])
        ic_a = nnan_consistency(a0, a1, corrtype=corrtype, ignore_vals=ignore_vals)
        ic_b = nnan_consistency(b0, b1, corrtype=corrtype, ignore_vals=ignore_vals)
        out['IC_a'].append(ic_a)
        out['IC_b'].append(ic_b)
        rho_tmp = []
        rho_tmp.append(nnan_consistency(a0, b0, corrtype=corrtype, ignore_vals=ignore_vals))
        rho_tmp.append(nnan_consistency(a1, b0, corrtype=corrtype, ignore_vals=ignore_vals))
        rho_tmp.append(nnan_consistency(a0, b1, corrtype=corrtype, ignore_vals=ignore_vals))
        rho_tmp.append(nnan_consistency(a1, b1, corrtype=corrtype, ignore_vals=ignore_vals))        
        out['rho'].append(np.mean(rho_tmp))
        out['rho_n'].append(np.mean(rho_tmp) / ((ic_a*ic_b)**0.5))
        out['rho_n_sq'].append(np.mean(rho_tmp)**2 / ((ic_a*ic_b)))

    return out
    