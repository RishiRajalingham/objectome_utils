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

def get_imgdata_from_trials_base(trials, meta):
    """ trials to imgdata (per split) """
    uobjs = list(set(meta['obj']))
    uimgs = [m['id'] for m in meta if m['obj'] in uobjs]
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

def get_imgdata_from_trials(trials, meta, niter=10):
    """ trials to imgdata """
    imgdata = []
    if isinstance(trials, list):
        niter = len(trials)
    for i in range(niter):
        if isinstance(trials, list): # trial splits are precomputed over the raw data (e.g. reps)
            id1 = get_imgdata_from_trials_base(trials[i][0], meta)
            id2 = get_imgdata_from_trials_base(trials[i][1], meta)
        else: # trials are split into random halves
            ntrials = trials.shape[0]
            tr = np.arange(ntrials)
            random.shuffle(tr)
            tr1 = tr[:int(len(tr)/2)]
            tr2 = tr[int(len(tr)/2):]
            id1 = get_imgdata_from_trials_base(trials[tr1], meta)
            id2 = get_imgdata_from_trials_base(trials[tr2], meta)

        imgdata_tmp = []
        imgdata_tmp.append(id1)
        imgdata_tmp.append(id2)
        imgdata.append(imgdata_tmp)
    return imgdata

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

def get_img_dprime_from_imgdata(imgdata, img_i, dist_obj_i=None):
    """ get dprime computed for image img_i, based on distracters dist_obj_i.
    can be used to compute both I1 (over any set of distracters) and I2.
    """
    if sum(imgdata['selection'][img_i,:]) == 0:
        return np.nan, np.nan  # img wasn't shown
    if dist_obj_i == None:
        obj_i = range(imgdata['true'].shape[1])
    else:
        lab = np.nonzero(imgdata['true'][img_i,:] > 0)[0][0]
        obj_i = np.unique([dist_obj_i, lab])
    
    if len(obj_i) == 1:
        return np.nan, np.nan  # distracter is label
    if (imgdata['selection'][img_i,dist_obj_i] == 0).all():
        return np.nan, np.nan  # img wasn't shown w these distracter

    nobjs = len(obj_i)
    true_ = imgdata['true'][:,obj_i]
    choice_ = imgdata['choice'][:,obj_i]
    selection_ = imgdata['selection'][:,obj_i]

    # what obj is img_i?
    t = np.nonzero(true_[img_i,:] > 0)[0][0]
    # all negative images
    t_i = np.nonzero(true_[:,t] == 0)[0] 

    # order positive + negatives images
    ind = np.concatenate(([img_i],t_i))
    # order positive + negative objs 
    ind2 = list(set(range(nobjs))-set([t]))
    ind2 = np.hstack(([t], ind2)) 

    hitmat_curr = choice_ / (1.0*selection_)
    hitmat_curr = hitmat_curr[ind,:]

    return dprime_from2x2(hitmat_curr[:,ind2])

def get_metric_from_imgdata_base(imgdata_s, compute_metrics={'I1_dprime', 'I2_dprime'}, rec_precomputed={}):
    """ imgdata to metrics (per split) """
    nimgs, nobjs = imgdata_s['true'].shape
    hitrates_ = imgdata_s['choice'] / (1.0*imgdata_s['selection'])
    hitrates_[imgdata_s['true'] > 0] = np.nan    
    
    rec = rec_precomputed

    if 'O2_dprime' in compute_metrics:
        rec['O2_dprime'] = get_o2_from_imgdata(imgdata_s)

    if 'I1_hitrate' in compute_metrics:
        rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
        rec['I1_hitrate_z'] = np.ones((nimgs,)) * np.nan
        rec['I1_hitrate_c'] = np.ones((nimgs,)) * np.nan
        rec['I1_dprime'] = np.ones((nimgs,)) * np.nan
        rec['I1_dprime_z'] = np.ones((nimgs,)) * np.nan
        rec['I1_dprime_c'] = np.ones((nimgs,)) * np.nan
        rec['I1_accuracy'] = np.ones((nimgs,)) * np.nan
        rec['I1_accuracy_z'] = np.ones((nimgs,)) * np.nan
        rec['I1_accuracy_c'] = np.ones((nimgs,)) * np.nan

    if 'I2_hitrate' in compute_metrics:
        rec['I2_hitrate'] = deepcopy(hitrates_)
        rec['I2_hitrate_z'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_hitrate_c'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_dprime'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_dprime_z'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_dprime_c'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_accuracy'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_accuracy_z'] = np.ones((nimgs,nobjs)) * np.nan
        rec['I2_accuracy_c'] = np.ones((nimgs,nobjs)) * np.nan

    for ii in range(nimgs):
        if 'I1_dprime' in compute_metrics:
            rec['I1_dprime'][ii],rec['I1_accuracy'][ii] = get_img_dprime_from_imgdata(imgdata_s, ii)
        if 'I2_dprime' in compute_metrics:
            for jj in range(nobjs):
                rec['I2_dprime'][ii,jj],rec['I2_accuracy'][ii,jj] = get_img_dprime_from_imgdata(imgdata_s, ii, jj)

    for ii in range(nobjs):
        t_i = np.nonzero(imgdata_s['true'][:,ii] > 0)[0]
        rec['I1_hitrate_z'][t_i] = nanzscore(rec['I1_hitrate'][t_i])
        rec['I1_hitrate_c'][t_i] = nanzscore(rec['I1_hitrate'][t_i], mean_only=True)
        if 'I1_dprime' in compute_metrics:
            rec['I1_dprime_z'][t_i] = nanzscore(rec['I1_dprime'][t_i])
            rec['I1_dprime_c'][t_i] = nanzscore(rec['I1_dprime'][t_i], mean_only=True)
            rec['I1_accuracy_z'][t_i] = nanzscore(rec['I1_accuracy'][t_i])
            rec['I1_accuracy_c'][t_i] = nanzscore(rec['I1_accuracy'][t_i], mean_only=True)

        if 'I2_dprime' in compute_metrics:
            for jj in range(nobjs):
                rec['I2_hitrate_z'][t_i,jj] = nanzscore(rec['I2_hitrate'][t_i,jj])
                rec['I2_hitrate_c'][t_i,jj] = nanzscore(rec['I2_hitrate'][t_i,jj], mean_only=True)
                rec['I2_dprime_z'][t_i,jj] = nanzscore(rec['I2_dprime'][t_i,jj])
                rec['I2_dprime_c'][t_i,jj] = nanzscore(rec['I2_dprime'][t_i,jj], mean_only=True)
                rec['I2_accuracy_z'][t_i,jj] = nanzscore(rec['I2_accuracy'][t_i,jj])
                rec['I2_accuracy_c'][t_i,jj] = nanzscore(rec['I2_accuracy'][t_i,jj], mean_only=True)
        
    return rec

def get_metric_from_imgdata(imgdata, compute_metrics={'I1_dprime'}, rec_precomputed={}):
    """ imgdata to metrics """    
    rec = rec_precomputed

    for fn in compute_metrics:
        rec[fn] = []
    for i in range(len(imgdata)):
        rec1 = get_metric_from_imgdata_base(imgdata[i][0], compute_metrics=compute_metrics, rec_precomputed=rec)
        rec2 = get_metric_from_imgdata_base(imgdata[i][1], compute_metrics=compute_metrics, rec_precomputed=rec)

        for fn in rec1.keys():
            rec_tmp = []
            rec_tmp.append(rec1[fn])
            rec_tmp.append(rec2[fn])
            rec[fn].append(rec_tmp)
    return rec

def compute_behavioral_metrics(trials, meta, compute_metrics={'I1_dprime'}, niter=10, imgdata=None, rec_precomputed={}):
    """ Main function to get metrics from trials """
    if imgdata == None:
        imgdata = get_imgdata_from_trials(trials, meta, niter=niter)

    rec = get_metric_from_imgdata(imgdata, compute_metrics=compute_metrics, rec_precomputed=rec_precomputed)

    return rec, imgdata

def subsample_imgdata(imgdata, img_i=None, obj_i=None):

    if img_i == None:
        img_i = range(imgdata[0][0]['true'].shape[0])
    if obj_i == None:
        obj_i = range(imgdata[0][0]['true'].shape[1])
    imgdata_sub = deepcopy(imgdata)
    for i in range(len(imgdata)):
        for j in range(2):
            for fn in imgdata[i][j].keys():
                tmp = imgdata[i][j][fn][img_i,:]
                imgdata_sub[i][j][fn] = tmp[:,obj_i]
    return imgdata_sub

""" Alternative methods for computing metrics directly from trials. Conversion to imgdata may be incorrect 
for I2/O2 computations, as it doesn't split trials into tasks first. """


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
    rec['I2_accuracy_c'][i,j] = rec['I2_accuracy'] - rec['O2_accuracy_exp']
    rec['I2_hitrate_c'][i,j] = rec['I2_hitrate'] - rec['O2_hitrate_exp']

    rec['I1_dprime'] = np.nanmean(rec['I2_dprime'], 1)
    rec['I1_accuracy'] = np.nanmean(rec['I2_accuracy'], 1)
    rec['I1_hitrate'] = np.nanmean(rec['I2_hitrate'], 1)

    rec['I1_dprime_c'] = np.nanmean(rec['I2_dprime_c'], 1)
    rec['I1_accuracy_c'] = np.nanmean(rec['I2_accuracy_c'], 1)
    rec['I1_hitrate_c'] = np.nanmean(rec['I2_hitrate_c'], 1)

    return rec

def get_metric_from_trials(trials, meta):
    rec = {}
    compute_metrics = [
        'O2_dprime', 'O2_accuracy', 'O2_hitrate', 
        'I2_dprime', 'I2_accuracy', 'I2_hitrate', 
        'I1_dprime', 'I1_accuracy', 'I1_hitrate', 
        'I2_dprime_c', 'I2_accuracy_c', 'I2_hitrate_c', 
        'I1_dprime_c', 'I1_accuracy_c', 'I1_hitrate_c'
        ]
    rec = dict.fromkeys(compute_metrics)
    for i in range(len(trials)):
        rec1 = get_metric_from_trials_base(trials[i][0], meta)
        rec2 = get_metric_from_trials_base(trials[i][1], meta)
        for fn in compute_metrics:
            rec[fn].append([rec1, rec2])
    return rec

def trials_to_metrics(trials, meta, niter=10):
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


