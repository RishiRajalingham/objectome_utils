import random
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from scipy.stats.mstats import zscore
from sklearn.metrics import confusion_matrix
from copy import deepcopy

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
    return np.clip(dp, -maxVal, maxVal)

def get_imgdata_from_trials_base(trials):
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
            id1 = get_imgdata_from_trials_base(trials[i][0])
            id2 = get_imgdata_from_trials_base(trials[i][1])
        else: # trials are split into random halves
            ntrials = trials.shape[0]
            tr = np.arange(ntrials)
            random.shuffle(tr)
            tr1 = tr[:int(len(tr)/2)]
            tr2 = tr[int(len(tr)/2):]
            id1 = get_imgdata_from_trials_base(trials[tr1])
            id2 = get_imgdata_from_trials_base(trials[tr2])

        imgdata_tmp = []
        imgdata_tmp.append(id1)
        imgdata_tmp.append(id2)
        imgdata.append(imgdata_tmp)

    return imgdata

def get_img_dprime_from_imgdata(imgdata, img_i, obj_i=None):
    """ get dprime computed for image img_i, based on distracters obj_i.
    can be used to compute both I1 (over any set of distracters) and I2.
    """
    if obj_i == None:
        obj_i = range(imgdata['true'].shape[1])
    else:
        lab = np.nonzero(imgdata['true'][img_i,:] > 0)[0][0]
        obj_i = np.unique([obj_i, lab])
        
    nobjs = len(obj_i)
    if nobjs == 1:
        return np.nan

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

def get_metric_from_imgdata_base(imgdata_s, compute_metrics={'I1_dprime'}):
    """ imgdata to metrics (per split) """
    nimgs, nobjs = imgdata_s['true'].shape
    hitrates_ = imgdata_s['choice'] / (1.0*imgdata_s['selection'])
    hitrates_[imgdata_s['true'] > 0] = np.nan
    rec = {}
    rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
    rec['I2_hitrate'] = hitrates_
    rec['I1_hitrate_z'] = 1.0 - np.nanmean(hitrates_,1)
    rec['I2_hitrate_z'] = hitrates_

    rec['I1_dprime'] = np.ones((nimgs,1)) * np.nan
    rec['I2_dprime'] = np.ones((nimgs,nobjs)) * np.nan
    rec['I1_dprime_z'] = np.ones((nimgs,1)) * np.nan
    rec['I2_dprime_z'] = np.ones((nimgs,nobjs)) * np.nan
    rec['I2_dprime_z2'] = np.ones((nimgs,nobjs)) * np.nan
    
    for ii in range(nimgs):
        if 'I2_dprime' in compute_metrics:
            rec['I1_dprime'][ii] = get_img_dprime_from_imgdata(imgdata_s, ii)
        if 'I2_dprime' in compute_metrics:
            for jj in range(nobjs):
                rec['I2_dprime'][ii,jj] = get_img_dprime_from_imgdata(imgdata_s, ii, jj)
        
    return rec

def get_metric_from_imgdata(imgdata, compute_metrics={'I1_dprime'}):
    """ imgdata to metrics """
    rec = {}
    for fn in compute_metrics:
        rec[fn] = []
    for i in range(len(imgdata)):
        rec1 = get_metric_from_imgdata_base(imgdata[i][0], compute_metrics=compute_metrics)
        rec2 = get_metric_from_imgdata_base(imgdata[i][1], compute_metrics=compute_metrics)

        for fn in compute_metrics:
            rec_tmp = []
            rec_tmp.append(rec1[fn])
            rec_tmp.append(rec2[fn])
            rec[fn].append(rec_tmp)
    return rec

def compute_behavioral_metrics(trials, meta, compute_metrics={'I1_dprime'}, niter=10, imgdata=None):
    """ Main function to get metrics from trials """
    rec_splithalves = {
        'O1_dprime':[],'O2_dprime':[],'I2_hitrate':[],'I2_dprime':[],
        'I1_hitrate':[],'I1_hitrate_c':[],'I1_hitrate_z':[],
        'I1_dprime':[],'I1_dprime_c':[],'I1_dprime_z':[]
    }

    if imgdata == None:
        imgdata = get_imgdata_from_trials(trials, meta, niter=niter)

    rec = get_metric_from_imgdata(imgdata, compute_metrics=compute_metrics)

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
    niter = min(len(A), len(B))
    out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[], 'rho_n_sq':[]}
    for i in range(niter):
        a0,a1 = A[metricn][i][0], A[metricn][i][1]
        b0,b1 = B[metricn][i][0], B[metricn][i][1]
        
        if img_subsample != None:
            a0,a1 = a0[img_subsample], a1[img_subsample]
            b0,b1 = b0[img_subsample], b1[img_subsample]
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




# def get_obj_dprime_from_trials(trials, SIGLABEL):
#     C2x2 = np.zeros((2,2))
#     C2x2[0,0] = sum((trials['sample_obj'] == SIGLABEL) & (trials['choice'] == SIGLABEL)) #hit
#     C2x2[0,1] = sum((trials['sample_obj'] == SIGLABEL) & (trials['choice'] != SIGLABEL)) # miss
#     C2x2[1,0] = sum((trials['sample_obj'] != SIGLABEL) & (trials['choice'] == SIGLABEL)) #false alarm
#     C2x2[1,1] = sum((trials['sample_obj'] != SIGLABEL) & (trials['choice'] != SIGLABEL)) # correct rej
#     return obj.dprime_from2x2(C2x2)






# def compute_behavioral_metrics_base(trials, meta, compute_metrics={'I1_dprime'}, O2_normalize=True, rec_precomputed=None):
#     uobjs = list(set(trials['sample_obj']))
#     uobjs = list(set(meta['obj']))
#     uimgs = [m['id'] for m in meta if m['obj'] in uobjs]

#     nimgs = len(uimgs)
#     nobjs = len(uobjs)

#     imgdata = get_confusion_patterns(trials, uobjs, uimgs)
#     imgdata_pertask = get_confusion_patterns_pertask(trials, uobjs, uimgs)
    
#     if rec_precomputed == None:
#         rec = {}
#     else:
#         compute_metrics = [cm for cm in compute_metrics if cm not in rec_precomputed.keys()]
#         rec = rec_precomputed

#     if ('O1_dprime' in compute_metrics) | ('O2_dprime' in compute_metrics):
#         rec['O1_dprime'] = np.ones((nobjs,1)) * np.nan
#         rec['O2_dprime'] = np.ones((nobjs,nobjs)) * np.nan
#         for ii in range(nobjs):
#             t0 = (trials['sample_obj'] == uobjs[ii])
#             t1 = (trials['dist_obj'] == uobjs[ii])
#             t = t0 | t1
#             rec['O1_dprime'][ii] = get_obj_dprime_from_trials(trials[t], uobjs[ii])
#             for jj in range(ii+1,nobjs):
#                 t0 = (trials['sample_obj'] == uobjs[ii]) & (trials['dist_obj'] == uobjs[jj])
#                 t1 = (trials['sample_obj'] == uobjs[jj]) & (trials['dist_obj'] == uobjs[ii])
#                 t = t0 | t1
#                 rec['O2_dprime'][ii,jj] = get_obj_dprime_from_trials(trials[t], uobjs[ii])
                
#     if ('I1_hitrate' in compute_metrics) | ('I2_hitrate' in compute_metrics):
#         hitrates_ = imgdata['choice'] / (1.0*imgdata['selection'])
#         rec['I2_hitrate'] = hitrates_
#         hitrates_[imgdata['true'] > 0] = np.nan
#         rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
        
#     if ('I1_dprime' in compute_metrics) | ('I2_dprime' in compute_metrics):
#         rec['I1_dprime'] = np.zeros((nimgs,1))
#         rec['I2_dprime'] = np.zeros((nimgs,nobjs))
#         for ii in range(nimgs):
#             if ('I1_dprime' in compute_metrics):
#                 rec['I1_dprime'] [ii,:] = get_img_dprime_from_imgdata(imgdata, ii)
#             if ('I2_dprime' in compute_metrics):
#                 for jj in range(nobjs):
#                     obj_i = np.nonzero(imgdata['true'][ii,:] > 0)[0]#what obj is this img
#                     if (len(obj_i) == 0):# | (obj_i[0] == jj):
#                         continue
#                     elif obj_i[0] == jj:
#                         continue
#                     else:
#                         obj_i = obj_i[0]
#                     rec['I2_dprime'][ii,jj] = get_img_dprime_from_imgdata(imgdata_pertask[obj_i][jj], ii)

#     if O2_normalize:
#         for fn in rec.keys():
#             if 'I1' in fn:
#                 metric = rec[fn]
#                 metric_z = np.zeros(metric.shape)
#                 metric_c = np.zeros(metric.shape)
#                 for jj in range(nobjs):
#                     t_i = np.nonzero(imgdata['true'][:,jj] > 0)[0]
#                     metric_z[t_i] = zscore(metric[t_i])
#                     metric_c[t_i] = metric[t_i] - np.mean(metric[t_i])
#                 rec[fn + '_z'] = metric_z
#                 rec[fn + '_c'] = metric_c
        
#     rec['uimgs'] = uimgs
#     rec['uobjs'] = uobjs


#     return rec, imgdata


# def recompute_behavioral_metrics_base(imgdata, rec={}, compute_metrics={'I1_dprime'}):
#     nimgs, nobjs = imgdata['choice'].shape

#     if ('I1_hitrate' in compute_metrics) | ('I2_hitrate' in compute_metrics):
#         hitrates_ = imgdata['choice'] / (1.0*imgdata['selection'])
#         rec['I2_hitrate'] = hitrates_
#         hitrates_[imgdata['true'] > 0] = np.nan
#         rec['I1_hitrate'] = 1.0 - np.nanmean(hitrates_,1)
        
#     if ('I1_dprime' in compute_metrics) | ('I2_dprime' in compute_metrics):
#         rec['I1_dprime'] = np.zeros((nimgs,1))
#         rec['I2_dprime'] = np.zeros((nimgs,nobjs))
#         for ii in range(nimgs):
#             if ('I1_dprime' in compute_metrics):
#                 rec['I1_dprime'] [ii,:] = get_img_dprime_from_imgdata(imgdata, ii)
#             if ('I2_dprime' in compute_metrics):
#                 for jj in range(nobjs):
#                     obj_i = np.nonzero(imgdata['true'][ii,:] > 0)[0]#what obj is this img
#                     if (len(obj_i) == 0):# | (obj_i[0] == jj):
#                         continue
#                     elif obj_i[0] == jj:
#                         continue
#                     else:
#                         obj_i = obj_i[0]
#                         rec['I2_dprime'][ii,jj] = get_img_dprime_from_imgdata(imgdata_pertask[obj_i][jj], ii)
#     return rec


# def recompute_behavioral_metrics():
#     import glob

#     fns = glob.glob('/mindhive/dicarlolab/u/rishir/monkey_objectome/behavioral_benchmark/data/splithalf_imgdata/*.pkl')
#     for fn in fns:
#         fn2 = fn.replace('splithalf_imgdata/', 'metrics/')
#         imgdata = pk.load(open(fn, 'r'))
#         rec = pk.load(open(fn2, 'r'))
#         for i in range(len(imgdata)):
#             for j in range(2):
#                 rec_new = recompute_behavioral_metrics_base(imgdata[i][j], rec={}, compute_metrics={'I1_dprime'})
#                 for rfn in rec_new.keys():
#                     rec[rfn][i][j] = rec_new[rfn]

#         with open(fn2, 'wb') as _f:
#             pk.dump(rec, _f)
#         print 'saved ' + fn2

        
