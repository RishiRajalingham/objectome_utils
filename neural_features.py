import tables as tbl
import numpy as np 
import cPickle as pk
import os 


STIMPATH = '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'
NEURALPATH = STIMPATH + 'neural_features/'
RAWDAT_fn = 'Tito_20130814_Objtt100sp01_S130404M96.trim.wh.evoked.repr.h5'
NIDX_fn = 'idx_TitoR.pkl'

meta = pk.load(open(STIMPATH + 'metadata.pkl'))
nidx = pk.load(open(NEURALPATH + NIDX_fn, 'r'))
rawdat = tbl.openFile(NEURALPATH + RAWDAT_fn)

iid = rawdat.root.meta.idx2iid.read()
M = rawdat.root.spk[0]

IT_mean = M[:,:,nidx['idx_TitoR_IT']].mean(0)
V4_mean = M[:,:,nidx['idx_TitoR_V4']].mean(0)

IT_rep = M[:,:,nidx['idx_TitoR_IT']]
V4_rep = M[:,:,nidx['idx_TitoR_V4']]


IT_features_dict = []
V4_features_dict = []
ITrep_features_dict = []
V4rep_features_dict = []


for i,imi in enumerate(iid):
	imi = imi.split('_')[-1].split('.')[-2]
	if imi in meta['id']:
		IT_features_dict.append({'id':imi, 
		'feature':IT_mean[i,:], 
		'feature_layer':'IT'
		})
		V4_features_dict.append({'id':imi, 
		'feature':V4_mean[i,:], 
		'feature_layer':'V4'
		})
		ITrep_features_dict.append({'id':imi, 
		'feature':IT_rep[:,i,:], 
		'feature_layer':'IT'
		})
		V4rep_features_dict.append({'id':imi, 
		'feature':V4_rep[:,i,:], 
		'feature_layer':'V4'
		})
		
if os.path.exists(NEURALPATH) == False:
    os.mkdir(NEURALPATH)
with open(NEURALPATH + 'V4.pkl', 'wb') as _f:
    pk.dump(V4_features_dict, _f)
with open(NEURALPATH + 'IT.pkl', 'wb') as _f:
   pk.dump(IT_features_dict, _f)
with open(NEURALPATH + 'V4_rep.pkl', 'wb') as _f:
    pk.dump(V4rep_features_dict, _f)
with open(NEURALPATH + 'IT_rep.pkl', 'wb') as _f:
   pk.dump(ITrep_features_dict, _f)


""" ic computations below"""
#f : images x reps x sites
import scipy.stats as stats
import numpy as np
niter = 10
IC = []
for iter in range(niter):
	inds = range(9)
	np.random.shuffle(inds)
	f1 = f[:,inds[:4],:].mean(1)
	f2 = f[:,inds[5:],:].mean(1)
	ic = [stats.pearsonr(f1[:,i], f2[:,i])[0] for i in range(141)]
	IC.append(ic)
IC = np.array(IC)

IC_m = IC.mean(0)

""" darren's IT data """


meta = pk.load(open(STIMPATH + 'metadata.pkl'))
nidx = pk.load(open(NEURALPATH + NIDX_fn, 'r'))
rawdat = tbl.openFile(NEURALPATH + RAWDAT_fn)

iid = rawdat.root.meta.idx2iid.read()
M = rawdat.root.spk[0]

IT_mean = M[:,:,nidx['idx_TitoR_IT']].mean(0)
V4_mean = M[:,:,nidx['idx_TitoR_V4']].mean(0)

IT_rep = M[:,:,nidx['idx_TitoR_IT']]
V4_rep = M[:,:,nidx['idx_TitoR_V4']]


IT_features_dict = []
V4_features_dict = []
ITrep_features_dict = []
V4rep_features_dict = []


for i,imi in enumerate(iid):
	imi = imi.split('_')[-1].split('.')[-2]
	if imi in meta['id']:
		IT_features_dict.append({'id':imi, 
		'feature':IT_mean[i,:], 
		'feature_layer':'IT'
		})
		V4_features_dict.append({'id':imi, 
		'feature':V4_mean[i,:], 
		'feature_layer':'V4'
		})
		ITrep_features_dict.append({'id':imi, 
		'feature':IT_rep[:,i,:], 
		'feature_layer':'IT'
		})
		V4rep_features_dict.append({'id':imi, 
		'feature':V4_rep[:,i,:], 
		'feature_layer':'V4'
		})
		
if os.path.exists(NEURALPATH) == False:
    os.mkdir(NEURALPATH)
with open(NEURALPATH + 'V4.pkl', 'wb') as _f:
    pk.dump(V4_features_dict, _f)
with open(NEURALPATH + 'IT.pkl', 'wb') as _f:
   pk.dump(IT_features_dict, _f)
with open(NEURALPATH + 'V4_rep.pkl', 'wb') as _f:
    pk.dump(V4rep_features_dict, _f)
with open(NEURALPATH + 'IT_rep.pkl', 'wb') as _f:
   pk.dump(ITrep_features_dict, _f)



"""  unprocessed spikes for hvm """
RAWDAT_dir = '/mindhive/dicarlolab/u/rishir/stimuli/hvm/neural_features/rawdat_unprocessed/'
RAWDAT_fn = ['Chabo_Tito_20110907_Var00a_pooled_P58.trim.repr.h5',
	'Chabo_Tito_20110907_Var03a_pooled_P58.trim.repr.h5',
	'Chabo_Tito_20110907_Var06a_pooled_P58.trim.repr.h5']

import tables as tbl
import numpy as np 
import cPickle as pk
import os 
import dldata.stimulus_sets.hvm as hvm

dset = hvm.HvMWithDiscfade()
meta = dset.meta
imglist = [m['filename'].split('/')[-1] for m in meta]
idlist = [m['id'].split('/')[-1] for m in meta]
IT_features_dict = []
V4_features_dict = []

IT_features_mean = np.zeros((len(meta), len(dset.IT_NEURONS)))
V4_features_mean = np.zeros((len(meta), len(dset.V4_NEURONS)))

for fn in RAWDAT_fn:
	rawdat = tbl.openFile(RAWDAT_dir + fn)
	M = rawdat.root.spk[0]
	meanFR = np.squeeze(np.mean(M,0))
	IT_mean = meanFR[:,dset.IT_NEURONS]
	V4_mean = meanFR[:,dset.V4_NEURONS]
	iid = rawdat.root.meta.idx2iid.read()

	for i,imi in enumerate(iid):
		if imi in imglist:
			ind = imglist.index(imi)
			id_ = idlist[ind]
			IT_features_dict.append({'id':id_, 
			'feature':IT_mean[i,:], 
			'feature_layer':'IT'
			})
			V4_features_dict.append({'id':id_, 
			'feature':V4_mean[i,:], 
			'feature_layer':'V4'
			})
			IT_features_mean[ind,:] = IT_mean[i,:]
			V4_features_mean[ind,:] = V4_mean[i,:]

feature_matrix = {'IT':IT_features_mean, 'V4':V4_features_mean}
NEURALPATH = '/mindhive/dicarlolab/u/rishir/stimuli/hvm/neural_features/'

with open(NEURALPATH + 'V4_unprocessed.pkl', 'wb') as _f:
    pk.dump(V4_features_dict, _f)
with open(NEURALPATH + 'IT_unprocessed.pkl', 'wb') as _f:
   pk.dump(IT_features_dict, _f)
with open(NEURALPATH + 'neural_feature_matrix_unprocessed.pkl', 'wb') as _f:
   pk.dump(feature_matrix, _f)

