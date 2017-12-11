import numpy as np 
import scipy as sp
from scipy import stats, linalg
from copy import deepcopy
""" General purpose utils """

def nanzscore(x, mean_only=False):
    if mean_only:
        y = np.ones(x.shape) * np.nan
        t = np.isnan(x) == False
        y[t] = x[t] - np.nanmean(x[t])
    else:
        zscore = sp.stats.mstats.zscore
        y = np.ones(x.shape) * np.nan
        t = np.isnan(x) == False
        y[t] = zscore(x[t])
    return y

def grpstats(X,G):
    mu, sig_se, sig_sd = [],[],[]
    UG = np.unique(G)
    for gi in UG:
        mu.append(np.nanmean(X[np.nonzero(G == gi)[0]]))
        x = X[np.nonzero(G == gi)[0]]
        x = np.array([i for i in x if not np.isnan(i)])
        sem = sp.stats.sem(x)
        sig_se.append(sem)
	sig_sd.append(np.nanstd(x))
    return UG, np.array(mu), np.array(sig_se), np.array(sig_sd)

def bin_variable(factor, prctile_step=10):
    bins = [np.percentile(factor,i) for i in range(0,100,prctile_step)]
    bins = [min(factor)] + bins + [max(factor)]
    factor_b = np.squeeze([np.nonzero(np.histogram(xi, bins)[0] == 1)[0] for xi in factor])
    factor_centers = grpstats(factor, factor_b)[1]
    return factor_b, factor_centers    




"""  Convert euler angles (in degrees, rx->ry->rz) to Quaternions """

def euler2q(rx0, ry0, rz0):
    import Quaternion as Q
    rx = rx0 * (np.pi / 180) / 2
    ry = ry0 * (np.pi / 180) / 2
    rz = rz0 * (np.pi / 180) / 2
    cz = np.cos(rz)
    sz = np.sin(rz)
    cy = np.cos(ry)
    sy = np.sin(ry)
    cx = np.cos(rx)
    sx = np.sin(rx)
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    w = cx*cy*cz + sx*sy*sz
    return Q.Quat(Q.normalize([x, y, z, w])).q

def get_quat_diff(q, q0):
    q = np.array(q)
    q0 = np.array(q0)
    dot = np.sum(q*q0)
    return np.arccos(abs(dot))

""" Linear reg - variance explained functions"""
import objectome_utils as obj 
from copy import deepcopy
from sklearn import linear_model


def lin_predict(x,y):
    x_mu = np.mean(x, axis=1)
    tnan = [t for t in range(len(y)) if (np.isfinite(x_mu[t]) and np.isfinite(y[t]))]
    regr = linear_model.LinearRegression()
    regr.fit(x[tnan],y[tnan])
    y_pred = deepcopy(y)
    y_pred[tnan] = regr.predict(x[tnan,:])
    out = {}
    out['y_pred'] = y_pred
    out['res'] = y - y_pred
    out['model'] = regr
    return out

def lin_predict_cv(x1, x2, y1, y2):
    out1 = lin_predict(x1,y1)
    out2 = lin_predict(x2,y2)

    out = {}
    ic_a = obj.nnan_consistency(x1, x2)
    ic_b = obj.nnan_consistency(y1, y2)
    rho = []
    rho.append(obj.nnan_consistency(y1, out2['y_pred']))
    rho.append(obj.nnan_consistency(y2, out1['y_pred']))
    out['IC_a'] = ic_a
    out['IC_b'] = ic_b
    out['rho'] = np.nanmean(rho)
    out['rho_n'] = np.nanmean(rho) / ((ic_a*ic_b)**0.5)
    out['rho_n_sq'] = np.nanmean(rho)**2 / ((ic_a*ic_b))    
    return out

# def lin_predict_cv(x, y1, y2):
#     t = np.squeeze(np.isfinite(y1))
#     x = np.squeeze(x[t,:])
#     y1 = np.squeeze(y1[t,:])
#     y2 = np.squeeze(y2[t,:])
#     regr = linear_model.LinearRegression()
#     regr.fit(x,y1)
#     return regr.predict(x), y2

def get_lin_variance_explained(x,y1,y2):
    z1,yy2 = lin_predict_cv(x,y1,y2)
    z2,yy1 = lin_predict_cv(x,y2,y1)
    
    ic1 = obj.nnan_consistency(yy1, yy2)
    ic2 = obj.nnan_consistency(z1, z2)
    rho1 = obj.nnan_consistency(yy1, z2)
    rho2 = obj.nnan_consistency(yy2, z1)
    
    return np.nanmean([rho1, rho2])**2 / (ic1*ic2)
    

def get_lin_variance_explained_v2(x1,x2,y1,y2):
    z1,yy2 = obj.lin_predict_cv(x1,y1,y2)
    z2,yy1 = obj.lin_predict_cv(x2,y2,y1)
    
    ic1 = obj.nnan_consistency(yy1, yy2)
    ic2 = obj.nnan_consistency(z1, z2)
    rho1 = obj.nnan_consistency(yy1, z2)
    rho2 = obj.nnan_consistency(yy2, z1)
    
    return np.nanmean([rho1, rho2])**2 / (ic1*ic2)
    
def unique_lin_variance(x,y1,y2,fi):
    varexp_all = get_lin_variance_explained(x,y1,y2)
    x2 = deepcopy(x)
    x2[:,fi] = 0
    varexp_exc = get_lin_variance_explained(x2,y1,y2)
    x3 = deepcopy(x)
    for i in range(x3.shape[1]):
        if i != fi:
            x3[:,i]  = 0
    varexp_only =  get_lin_variance_explained(x3,y1,y2)
    return varexp_all, varexp_exc, varexp_only

""" partial corr (from fabianp)"""



def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr


    """ dictionary utils """

def concatenate_dictionary(dict_tuple, fns=None):
    concat_dict = {}
    if fns == None:
        fns = dict_tuple[0].keys()
    for fn in fns:
        concat_dict[fn] = []
        for dict_ in dict_tuple:
            concat_dict[fn] = concat_dict[fn] + list(dict_[fn])
    return concat_dict

def subsample_dictionary(dict_in, inds):
    dkeys = dict_in.keys()
    dict_out = {}
    inds = np.array(inds)
    for dk in dkeys:
        dict_out[dk] = dict_in[dk][inds]
    return dict_out
    


def get_numbered_field(attribute):
    attr_idxs = {}            
    attr_index = []
    unique_attr = list(set(attribute))
    for idx, attr in enumerate(unique_attr):
        attr_idxs[attr] = idx
    for at in attribute:
        attr_index.append(attr_idxs[at])
    return unique_attr, attr_index
