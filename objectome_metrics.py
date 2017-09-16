import random
import copy
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from objectome_utils import nanzscore

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
    """ From a set of trials, get logical indexing for task conditions (sample image/object, distracter object, etc... """
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

    image_summary =[{'id':[]} for i in range(nimgs)]
    for i,OI in enumerate(uimgs):
        image_summary[i]['id'] = (trials['id'] == OI)
    return use_trial_samples, summary, image_summary

def get_c2x2_from_logicals(Si, Di, Ci=None, trials=None):
    cont_table = np.zeros((2,2))
    if Ci is not None:
        Si_i = np.logical_and(Si, Ci)
        Si_ni = np.logical_and(Si, np.logical_not(Ci))
        Di_i = np.logical_and(Di, Ci)
        Di_ni = np.logical_and(Di, np.logical_not(Ci))
        cont_table[0,0] = np.sum(Si_i)
        cont_table[0,1] = np.sum(Si_ni)
        cont_table[1,0] = np.sum(Di_i)
        cont_table[1,1] = np.sum(Di_ni) 
    elif trials is not None:
        xi = np.array(trials['prob_choice'][Si]).astype('double')
        xj = np.array(trials['prob_choice'][Di]).astype('double')
        cont_table[0,0] = np.nanmean(xi)
        cont_table[0,1] = 1-np.nanmean(xi)
        cont_table[1,0] = 1-np.nanmean(xj)
        cont_table[1,1] = np.nanmean(xj)
    return cont_table

def get_object_c2x2(trials, summary, obj_i, obj_j=None, use_trial_samples=False):
    """ Get 2x2 object level contingency table for object i, distracter j """
    Si, Di = summary[obj_i]['sam'], summary[obj_i]['dist']
    if obj_j is not None:
        Si = np.logical_and(Si, summary[obj_j]['dist'])
        Di = np.logical_and(Di, summary[obj_j]['sam'])

    if use_trial_samples:
        Ci = summary[obj_i]['choice']
        return get_c2x2_from_logicals(Si, Di, Ci=Ci)
    else:
        return get_c2x2_from_logicals(Si, Di, trials=trials)

def get_image_c2x2(trials, summary, image_summary, img_i, obj_i, obj_j=None, use_trial_samples=False):
    cont_table = np.zeros((2,2))
    Si = image_summary[img_i]['id']
    Di = summary[obj_i]['dist']
    
    if obj_j is not None:
        Si = np.logical_and(Si, summary[obj_j]['dist'])
        Di = np.logical_and(Di, summary[obj_j]['sam'])

    if use_trial_samples:
        Ci = summary[obj_i]['choice']
        return get_c2x2_from_logicals(Si, Di, Ci=Ci)
    else:
        return get_c2x2_from_logicals(Si, Di, trials=trials)

def normalize_by_object(metric, img_ind_per_obj):
    metric_n = copy.deepcopy(metric)
    for ind_i in img_ind_per_obj:
        for j in range(metric.shape[1]):
            tmp = metric[ind_i,j]
            mu = np.nanmean(tmp)
            for ii in ind_i:
                metric_n[ii,j] = (metric[ii,j] - mu)
    return metric_n                

def get_metric_base(trials, meta, compute_O=True, compute_I=True):

    uobjs, uimgs = list(set(meta['obj'])), list(meta['id'])
    nobjs, nimgs = len(uobjs), len(uimgs)
    use_trial_samples, summary, image_summary = trial_logicals(trials, meta)
    dprime_maxval = 5
    # if use_trial_samples:
    #     dprime_maxval = 10
    
    rec = {}
    if compute_O:
        # object level metrics
        rec['O1_dprime'] = np.ones((nobjs, 1)) * np.nan
        rec['O1_accuracy'] = np.ones((nobjs, 1)) * np.nan
        rec['O1_hitrate'] = np.ones((nobjs, 1)) * np.nan

        rec['O2_dprime'] = np.ones((nobjs, nobjs)) * np.nan
        rec['O2_accuracy'] = np.ones((nobjs, nobjs)) * np.nan
        rec['O2_hitrate'] = np.ones((nobjs, nobjs)) * np.nan
        
        for i in range(nobjs):   
            o1_2x2 = get_object_c2x2(trials, summary, i, obj_j=None, use_trial_samples=use_trial_samples)
            dp,ba,hr,cr = dprime_from2x2(o1_2x2, maxVal=dprime_maxval)
            rec['O1_dprime'][i,0] = dp
            rec['O1_accuracy'][i,0] = ba
            rec['O1_hitrate'][i,0] = hr
            
            for j in range(i+1, nobjs):
                o2_2x2 = get_object_c2x2(trials, summary, i, obj_j=j, use_trial_samples=use_trial_samples)
                dp,ba,hr,cr = dprime_from2x2(o2_2x2, maxVal=dprime_maxval)
                rec['O2_dprime'][i,j] = dp
                rec['O2_accuracy'][i,j] = ba
                rec['O2_hitrate'][i,j] = hr
                rec['O2_hitrate'][j,i] = cr

        rec['O1_dprime_v2'] = np.nanmean(rec['O2_dprime'], 1)
        rec['O1_dprime_v2'] = np.reshape(rec['O1_dprime_v2'], (rec['O1_dprime_v2'].shape[0], 1))

    if compute_I:
        # image level metrics
        rec['I1_dprime']  = np.ones((nimgs, 1)) * np.nan
        rec['I1_accuracy']  = np.ones((nimgs, 1)) * np.nan
        rec['I1_hitrate']  = np.ones((nimgs, 1)) * np.nan

        rec['I2_dprime']  = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_accuracy']  = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_hitrate']  = np.ones((nimgs,nobjs)) * np.nan

        for i in range(nimgs):
            obj_i = uobjs.index(meta[meta['id'] == uimgs[i]]['obj'])
            i1_2x2 = get_image_c2x2(trials, summary, image_summary, i, obj_i, obj_j=None, use_trial_samples=use_trial_samples)
            dp,ba,hr,cr = dprime_from2x2(i1_2x2, maxVal=dprime_maxval)
            rec['I1_dprime'][i,0] = dp
            rec['I1_accuracy'][i,0] = ba
            rec['I1_hitrate'][i,0] = hr

            for j in range(nobjs):
                i2_2x2 = get_image_c2x2(trials, summary, image_summary, i, obj_i, obj_j=j, use_trial_samples=use_trial_samples)
                dp,ba,hr,cr = dprime_from2x2(i2_2x2, maxVal=dprime_maxval)
                rec['I2_dprime'][i,j] = dp
                rec['I2_accuracy'][i,j] = ba
                rec['I2_hitrate'][i,j] = hr

        rec['I1_dprime_v2'] = np.nanmean(rec['I2_dprime'], 1)
        rec['I1_dprime_v2'] = np.reshape(rec['I1_dprime_v2'], (rec['I1_dprime_v2'].shape[0], 1))

        # normalizations
        img_ind_per_obj = []
        for uo in uobjs:
            tid = meta[meta['obj'] == uo]['id']
            ind = [uimgs.index(ti) for ti in tid]
            img_ind_per_obj.append(ind)

        rec['I1_dprime_C'] = normalize_by_object(rec['I1_dprime'], img_ind_per_obj)
        rec['I1_dprime_v2_C'] = normalize_by_object(rec['I1_dprime_v2'], img_ind_per_obj)
        rec['I2_dprime_C'] = normalize_by_object(rec['I2_dprime'], img_ind_per_obj)

    return rec

def compute_behavioral_metrics(trials, meta, niter, compute_O=True, compute_I=True, noise_model='trial_samples'):
    metrics = [
        'O1_dprime', 'O1_accuracy', 'O1_hitrate', 'O1_dprime_v2',
        'I1_dprime', 'I1_accuracy', 'I1_hitrate', 'I1_dprime_v2', 
        'O2_dprime', 'O2_accuracy', 'O2_hitrate', 
        'I2_dprime', 'I2_accuracy', 'I2_hitrate', 
        'I1_dprime_C', 'I1_dprime_v2_C', 'I2_dprime_C'
        ]

    if noise_model == None:
        # run on all trials just once without any sampling
        rec_all = get_metric_base(trials, meta, compute_O=compute_O, compute_I=compute_I)
    rec = {k: [] for k in metrics}
    
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
            rec1 = get_metric_base(trials[tr1], meta, compute_O=compute_O, compute_I=compute_I)
            rec2 = get_metric_base(trials[tr2], meta, compute_O=compute_O, compute_I=compute_I)
            
        for fn in metrics:
            if (fn in rec1) & (fn in rec2):
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

def pairwise_consistency(A, B, metricn='I1_dprime_C', corrtype='pearson', img_subsample=None, ignore_vals=None):
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
    