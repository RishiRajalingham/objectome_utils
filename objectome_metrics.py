import random
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from objectome_utils import nanzscore


""" 
Basic metric computations from aggregated trials to random split-halves of 
trials, image data (summary data structure), and metrics.
"""

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

def get_trial_splithalves(trials,niter=10):
    trial_splithalves = []
    for i in range(niter):
        ntrials = trials.shape[0]
        tr = np.arange(ntrials)
        random.shuffle(tr)
        tr1 = tr[:int(len(tr)/2)]
        tr2 = tr[int(len(tr)/2):]
        tsh_tmp = []
        tsh_tmp.append(trials[tr1])
        tsh_tmp.append(trials[tr2])
        trial_splithalves.append(tsh_tmp)

    return trial_splithalves

def get_metric_from_trials_base(trials, meta):

    uobjs = list(set(meta['obj']))
    nobj = len(uobjs)
    uimgs = [m['id'] for m in meta if m['obj'] in uobjs]
    nimgs = len(uimgs)

    rec = {}
    
    rec['O2_dprime'] = np.ones((nobj, nobj)) * np.nan
    rec['O2_accuracy'] = np.ones((nobj, nobj)) * np.nan
    rec['O2_hitrate'] = np.ones((nobj, nobj)) * np.nan
    
    rec['O2_dprime_exp'] = np.ones((nimgs, nobj)) * np.nan
    rec['O2_accuracy_exp'] = np.ones((nimgs, nobj)) * np.nan
    rec['O2_hitrate_exp'] = np.ones((nimgs, nobj)) * np.nan
    

    rec['I2_dprime']  = np.ones((nimgs,nobj)) * np.nan
    rec['I2_accuracy']  = np.ones((nimgs,nobj)) * np.nan
    rec['I2_hitrate']  = np.ones((nimgs,nobj)) * np.nan
    rec['I2_dprime_c']  = np.ones((nimgs,nobj)) * np.nan
    rec['I2_accuracy_c']  = np.ones((nimgs,nobj)) * np.nan
    rec['I2_hitrate_c']  = np.ones((nimgs,nobj)) * np.nan

    for i in range(nobj):

        OI = uobjs[i]
        sam_i = (trials['sample_obj'] == OI)
        dist_i = (trials['dist_obj'] == OI)
        choice_i = (trials['choice'] == OI)
        img_ind_i = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OI) ])
        
        for j in range(i+1, nobj):
            OJ = uobjs[j]
            sam_j = (trials['sample_obj'] == OJ)
            dist_j = (trials['dist_obj'] == OJ)
            choice_j = (trials['choice'] == OJ)
            img_ind_j = np.array([uimgs.index(m['id']) for m in meta if (m['obj'] == OJ) ])
            s_i = sam_i & dist_j
            s_j = sam_j & dist_i
            
            cont_table = np.zeros((2,2))
            cont_table[0,0] = sum(s_i & choice_i)
            cont_table[0,1] = sum(s_i & choice_j)
            cont_table[1,0] = sum(s_j & choice_i)
            cont_table[1,1] = sum(s_j & choice_j)
            dp,ba,hr,cr = dprime_from2x2(cont_table)
            rec['O2_dprime'][i,j] = dp
            rec['O2_accuracy'][i,j] = ba
            rec['O2_hitrate'][i,j] = hr
            rec['O2_hitrate'][j,i] = cr

            rec['O2_dprime_exp'][img_ind_i,:][:,j] = dp
            rec['O2_dprime_exp'][img_ind_j,:][:,i] = dp
            rec['O2_hitrate_exp'][img_ind_i,:][:,j] = hr
            rec['O2_hitrate_exp'][img_ind_j,:][:,i] = cr
            rec['O2_accuracy_exp'][img_ind_i,:][:,j] = ba
            rec['O2_accuracy_exp'][img_ind_j,:][:,i] = ba
            
        for j in range(nimgs):
            t_j = trials['id'] == uimgs[j] 
            OJ = meta[meta['id'] == uimgs[j]]['obj']
            s_obj_j = uobjs.index(OJ)
            if s_obj_j == i:
                continue
            
            dist_j = (trials['dist_obj'] == OJ)
            choice_j = (trials['choice'] == OJ)
            
            s_j = t_j & dist_i
            s_i = sam_i & dist_j 
            
            cont_table = np.zeros((2,2))
            cont_table[0,0] = sum(s_j & choice_j)
            cont_table[0,1] = sum(s_j & choice_i)
            cont_table[1,0] = sum(s_i & choice_j)
            cont_table[1,1] = sum(s_i & choice_i)

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

def get_metric_from_trials(trials, meta):
    compute_metrics = [
        'O2_dprime', 'O2_accuracy', 'O2_hitrate', 
        'I2_dprime', 'I2_accuracy', 'I2_hitrate', 
        'I1_dprime', 'I1_accuracy', 'I1_hitrate', 
        'I2_dprime_c', 'I2_accuracy_c', 'I2_hitrate_c', 
        'I1_dprime_c', 'I1_accuracy_c', 'I1_hitrate_c'
        ]
    rec = {k: [] for k in compute_metrics}
    for i in range(len(trials)):
        rec1 = get_metric_from_trials_base(trials[i][0], meta)
        rec2 = get_metric_from_trials_base(trials[i][1], meta)
        for fn in compute_metrics:
            rec[fn].append([rec1, rec2])
    return rec

def compute_behavioral_metrics(trials, meta, niter=10):
    # main call
    trial_sh = get_trial_splithalves(trials,niter=niter)
    rec = get_metric_from_trials(trial_sh, meta)
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

def pairwise_consistency_perobj(A,B, metricn, uobj, corrtype='pearson'):
    niter = min(len(A), len(B))
    out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[], 'rho_n_sq':[]}
    for i in range(niter):
        IC_a, IC_b, RHO, RHO_N, RHO_N_SQ = [],[],[],[],[]
        for obj_oi in uobj:
            win = meta['obj'] == obj_oi
            a0,a1 = A[metricn][i][0][win], A[metricn][i][1][win]
            b0,b1 = B[metricn][i][0][win], B[metricn][i][1][win]

            ic_a = nnan_consistency(a0,a1, corrtype)
            ic_b = nnan_consistency(b0,b1, corrtype)
            rho_tmp = [];
            rho_tmp.append(nnan_consistency(a0,b0, corrtype))
            rho_tmp.append(nnan_consistency(a1,b0, corrtype))
            rho_tmp.append(nnan_consistency(a0,b1, corrtype))
            rho_tmp.append(nnan_consistency(a1,b1, corrtype))

            rho = np.mean(rho_tmp)
            rho_n = np.mean(rho_tmp) / ((ic_a*ic_b)**0.5)
            rho_n_sq = np.mean(rho_tmp)**2 / ((ic_a*ic_b))
                                     
            IC_a.append(ic_a)
            IC_b.append(ic_b)
            RHO.append(rho)
            RHO_N.append(rho_n)
            RHO_N_SQ.append(rho_n_sq)
                                     
        out['IC_a'].append(IC_a)
        out['IC_b'].append(IC_b)
        out['rho'].append(RHO)
        out['rho_n'].append(RHO_N)
        out['rho_n_sq'].append(RHO_N_SQ)
    return out

def pairwise_consistency(A,B, metricn='I1_dprime_z', corrtype='pearson', img_subsample=None):
    niter = min(len(A[metricn]), len(B[metricn]))
    out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[], 'rho_n_sq':[]}
    for i in range(niter):
        a0,a1 = A[metricn][i][0], A[metricn][i][1]
        b0,b1 = B[metricn][i][0], B[metricn][i][1]
        
        if img_subsample != None:
            if a0.ndim == 1:
                a0,a1 = a0[img_subsample], a1[img_subsample]
                b0,b1 = b0[img_subsample], b1[img_subsample]
            else:
                a0,a1 = a0[img_subsample,:], a1[img_subsample,:]
                b0,b1 = b0[img_subsample,:], b1[img_subsample,:]
        ind = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(b0) & np.isfinite(b1)
        a0 = np.squeeze(a0[ind])
        a1 = np.squeeze(a1[ind])
        b0 = np.squeeze(b0[ind])
        b1 = np.squeeze(b1[ind])

        ic_a = nnan_consistency(a0,a1, corrtype)
        ic_b = nnan_consistency(b0,b1, corrtype)
        out['IC_a'].append(ic_a)
        out['IC_b'].append(ic_b)

        rho_tmp = [];
        rho_tmp.append(nnan_consistency(a0,b0, corrtype))
        rho_tmp.append(nnan_consistency(a1,b0, corrtype))
        rho_tmp.append(nnan_consistency(a0,b1, corrtype))
        rho_tmp.append(nnan_consistency(a1,b1, corrtype))
        
        out['rho'].append(np.mean(rho_tmp))
        out['rho_n'].append( np.mean(rho_tmp) / ((ic_a*ic_b)**0.5) )
        out['rho_n_sq'].append( np.mean(rho_tmp)**2 / ((ic_a*ic_b)) )

    return out


