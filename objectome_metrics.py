import random
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

def dprime_from2x2(C):
    """ Input matrix C is essentially a 2x2 confusion matrix, 
    rows and columsn: [A| !A], but !A can be a vector"""
    maxVal = 5
    hr_ = C[0,0] / (1.0*np.nansum(C[0,:]))
    fp_ = np.nansum(C[1:,0]) / (1.0*np.nansum(C[1:,:]))
    dp = norm.ppf(hr_,0,1) - norm.ppf(fp_,0,1)
    dprime = np.clip(dp, -maxVal, maxVal)
    balacc = 0.5*(hr_ + (1-fp_))
    hitrate = hr_
    corr_rej = 1-fp_
    return dprime, balacc, hitrate, corr_rej

def get_metric_from_probs_base(trials, meta, metric_spec='all'):

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
            cont_table[0,0] = np.nanmean(xi)
            cont_table[0,1] = 1 - np.nanmean(xi)
            cont_table[1,0] = 1 - np.nanmean(xj)
            cont_table[1,1] = np.nanmean(xj)
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
        summary[i]['sam'] = (trials['sample_obj'] == OI)
        summary[i]['dist'] = (trials['dist_obj'] == OI)
        summary[i]['choice'] = (trials['choice'] == OI)
        summary[i]['imgind'] = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OI) ])

    for i in range(nobjs):   
        img_ind_i = summary[i]['imgind']
        for j in range(i+1, nobjs):
            img_ind_j = summary[j]['imgind']

            cont_table = np.zeros((2,2))
            # read this next line as trials with: sample I, distracter J, choice I.
            cont_table[0,0] = np.sum(summary[i]['sam'] & summary[j]['dist'] & summary[i]['choice'])
            cont_table[0,1] = np.sum(summary[i]['sam'] & summary[j]['dist'] & summary[j]['choice'])
            cont_table[1,0] = np.sum(summary[j]['sam'] & summary[i]['dist'] & summary[i]['choice'])
            cont_table[1,1] = np.sum(summary[j]['sam'] & summary[i]['dist'] & summary[j]['choice'])
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
            t_j = trials['id'] == uimgs[j] 
            OJ = meta[meta['id'] == uimgs[j]]['obj']
            j_ = uobjs.index(OJ)
            if j_ == i:
                continue
            cont_table = np.zeros((2,2))
            # read this next line as trials with: sample j, distracter I, choice J.
            cont_table[0,0] = np.sum(t_j & summary[i]['dist'] & summary[j_]['choice'])
            cont_table[0,1] = np.sum(t_j & summary[i]['dist'] & summary[i]['choice'])
            cont_table[1,0] = np.sum(summary[i]['sam'] & summary[j_]['dist'] & summary[j_]['choice'])
            cont_table[1,1] = np.sum(summary[i]['sam'] & summary[j_]['dist'] & summary[i]['choice'])

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

def get_metric_base(trials, meta, prob_estimate=False, metric_spec='all'):
    if prob_estimate:
        rec = get_metric_from_probs_base(trials, meta, metric_spec=metric_spec)
    else:
        rec = get_metric_from_trials_base(trials, meta, metric_spec=metric_spec)
    return rec

def compute_behavioral_metrics(trials, meta, niter, prob_estimate=False, metric_spec='all', noise_model='trial_samples'):
    metrics = [
        'O2_dprime', 'O2_accuracy', 'O2_hitrate', 
        'I2_dprime', 'I2_accuracy', 'I2_hitrate', 
        'I1_dprime', 'I1_accuracy', 'I1_hitrate', 
        'I2_dprime_c', 'I2_accuracy_c', 'I2_hitrate_c', 
        'I1_dprime_c', 'I1_accuracy_c', 'I1_hitrate_c'
        ]

    if noise_model == None:
        # run on all trials just once without any sampling
        rec_all = get_metric_base(trials, meta, prob_estimate=prob_estimate, metric_spec=metric_spec)

    rec = {k: [] for k in metrics}
    
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
            rec1 = get_metric_base(trials[tr1], meta, prob_estimate=prob_estimate, metric_spec=metric_spec)
            rec2 = get_metric_base(trials[tr2], meta, prob_estimate=prob_estimate, metric_spec=metric_spec)
        for fn in metrics:
            rec[fn].append([rec1[fn], rec2[fn]])
    return rec
     

"""
Methods for measuring consistency of output metric
"""
def get_mean_behavior(b1, metricn):
    return np.squeeze(np.nanmean(np.nanmean(b1[metricn], axis=1), axis=0))

def nnan_consistency(A,B, corrtype='pearson'):
    ind = np.isfinite(A) & np.isfinite(B)
    if corrtype == 'pearson':
        return pearsonr(A[ind], B[ind])[0]
    elif corrtype == 'spearman':
        return spearmanr(A[ind], B[ind])[0]

def pairwise_consistency(A,B, metricn='I1_dprime_z', corrtype='pearson', img_subsample=None):
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
        ic_a = nnan_consistency(a0, a1, corrtype)
        ic_b = nnan_consistency(b0, b1, corrtype)
        out['IC_a'].append(ic_a)
        out['IC_b'].append(ic_b)
        rho_tmp = []
        rho_tmp.append(nnan_consistency(a0, b0, corrtype))
        rho_tmp.append(nnan_consistency(a1, b0, corrtype))
        rho_tmp.append(nnan_consistency(a0, b1, corrtype))
        rho_tmp.append(nnan_consistency(a1, b1, corrtype))        
        out['rho'].append(np.mean(rho_tmp))
        out['rho_n'].append(np.mean(rho_tmp) / ((ic_a*ic_b)**0.5))
        out['rho_n_sq'].append(np.mean(rho_tmp)**2 / ((ic_a*ic_b)))

    return out
    