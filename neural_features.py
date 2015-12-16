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