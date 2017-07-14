import numpy as np 
import scipy as sp


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



# Convert euler angles (in degrees, rx->ry->rz) to Quaternions
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

# # compute quaternion difference
# def get_quat_diff(q, q0):
#     import Quaternion as Q
#     Qq = Q.Quat(q)
#     Qq0 = Q.Quat(q0)
#     Qdiff = Qq * Qq0.inv()
#     return np.arccos(np.abs(Qdiff.q))





def features_to_metrics(features, meta, suffix=''):
    data = np.load(featurespath + f + '.npy')
    all_features[f] = data
    all_metas[f] = meta
    trials = obj.testFeatures(all_features, all_metas, fnames, obj.models_combined24)
    outfn = datapath + 'trials/' + f + suffix + '.pkl'
    pk.dump(trials, open(outfn, 'wb'))
    print '%10.2f seconds' % (time.time() - t)
    print 'saved ' + outfn
    return

def trials_to_metrics(trials_dict, compute_metrics={'I1_dprime','I1_hitrate','I2_hitrate', 'I2_dprime', 'O1_dprime','O2_dprime'}, suffix=''):
    for f in trials_dict:
        t = time.time()
        metrics, imgdata = obj.compute_behavioral_metrics(trials_dict[f], meta, compute_metrics=compute_metrics, O2_normalize=True)
        outfn = datapath + 'metrics/' + suffix + f + '_objectome24.pkl'
        outfn_2 = datapath + 'splithalf_imgdata/' + suffix + f + '_objectome24.pkl'
        # save_metrics(metrics, outfn)
        pk.dump(metrics, open(outfn, 'wb'))
        pk.dump(imgdata, open(outfn_2, 'wb'))
        print '%10.2f seconds' % (time.time() - t)
        print 'saved ' + outfn
    return


# def pairwise_consistency(A,B, metricn):
#     niter = min(len(A), len(B))
#     out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[]}
#     for i in range(niter):
#         a0,a1 = A[metricn][i][0], A[metricn][i][1]
#         b0,b1 = B[metricn][i][0], B[metricn][i][1]
        
#         ic_a = nnan_consistency(a0,a1)
#         ic_b = nnan_consistency(b0,b1)
#         out['IC_a'].append(ic_a)
#         out['IC_b'].append(ic_b)

#         rho_tmp = [];
#         rho_tmp.append(nnan_consistency(a0,b0))
#         rho_tmp.append(nnan_consistency(a1,b0))
#         rho_tmp.append(nnan_consistency(a0,b1))
#         rho_tmp.append(nnan_consistency(a1,b1))
        
#         out['rho'].append(np.mean(rho_tmp))
#         out['rho_n'].append( np.mean(rho_tmp) / ((ic_a*ic_b)**0.5) )

#     return out

# def pairwise_consistency(A,B, metricn):
#     niter = min(len(A), len(B))
#     out = {'IC_a':[], 'IC_b':[], 'rho':[], 'rho_n':[]}
#     for i in range(niter):
#         a0,a1 = A[i][0][metricn], A[i][1][metricn]
#         b0,b1 = B[i][0][metricn], B[i][1][metricn]
        
#         ic_a = nnan_consistency(a0,a1)
#         ic_b = nnan_consistency(b0,b1)
#         out['IC_a'].append(ic_a)
#         out['IC_b'].append(ic_b)

#         rho_tmp = [];
#         rho_tmp.append(nnan_consistency(a0,b0))
#         rho_tmp.append(nnan_consistency(a1,b0))
#         rho_tmp.append(nnan_consistency(a0,b1))
#         rho_tmp.append(nnan_consistency(a1,b1))
        
#         out['rho'].append(np.mean(rho_tmp))
#         out['rho_n'].append( np.mean(rho_tmp) / ((ic_a*ic_b)**0.5) )

#     return out


def nnan_consistency(A,B, corrtype='pearson'):
    ind = np.isfinite(A) & np.isfinite(B)
    if corrtype == 'pearson':
        return sp.stats.pearsonr(A[ind], B[ind])[0]
    elif corrtype == 'spearman':
        return sp.stats.spearmanr(A[ind], B[ind])[0]

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
