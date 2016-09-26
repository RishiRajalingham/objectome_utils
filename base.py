
""" Lists of subsets of objectome objects used for various experiments. """

models_objectome64 = ['weimaraner', 'lo_poly_animal_TRTL_B', 'lo_poly_animal_ELE_AS1', 'lo_poly_animal_TRANTULA', 'foreign_cat', 'lo_poly_animal_CHICKDEE', 'lo_poly_animal_HRS_ARBN',
 	'MB29346', 'MB31620', 'MB29874', 'interior_details_033_2', 'MB29822', 'face7', 'single_pineapple', 'pumpkin_3', 'Hanger_02', 'MB31188', 'antique_furniture_item_18', 'MB27346',
 	'interior_details_047_1', 'laptop01', 'womens_stockings_01M', 'pear_obj_2', 'household_aid_29', '22_acoustic_guitar', 'MB30850', 'MB30798', 'MB31015', 'Nurse_pose01', 'fast_food_23_1',
 	'kitchen_equipment_knife2', 'flarenut_spanner', 'womens_halterneck_06', 'dromedary', 'MB30758', 'MB30071', 'leaves16', 'lo_poly_animal_DUCK', '31_african_drums', 'lo_poly_animal_RHINO_2',
 	'lo_poly_animal_ANT_RED', 'interior_details_103_2', 'interior_details_103_4', 'MB27780', 'MB27585', 'build51', 'Colored_shirt_03M', 'calc01', 'Doctor_pose02', 'bullfrog', 'MB28699',
 	'jewelry_29', 'trousers_03', '04_piano', 'womens_shorts_01M', 'womens_Skirt_02M', 'lo_poly_animal_TIGER_B', 'MB31405', 'MB30203', 'zebra', 'lo_poly_animal_BEAR_BLK', 'lo_poly_animal_RB_TROUT',
 	'interior_details_130_2', 'Tie_06']

models_training8 = ['flarenut_spanner', 'lo_poly_animal_RHINO_2', 'womens_stockings_01M', 
 'dromedary', 'MB30758', 'lo_poly_animal_CHICKDEE', 'womens_shorts_01M', 'lo_poly_animal_ELE_AS1']

models_testing8_b1 =  ['build51', 'Hanger_02', 'interior_details_130_2', 'MB28699',
'zebra', 'interior_details_103_4', '22_acoustic_guitar', 'lo_poly_animal_BEAR_BLK']

models_testing8_b2 = ['MB30203', 'weimaraner', 'lo_poly_animal_TRANTULA', 'MB29874',
'antique_furniture_item_18', 'calc01', 'MB27346', 'kitchen_equipment_knife2'] 

models_testing8_b3 = ['Tie_06', 'leaves16', 'kitchen_equipment_knife2', 'foreign_cat', 
'womens_halterneck_06', 'womens_Skirt_02M', 'MB29346', 'MB29822'] # not used on monkeys (behavior & phys)

models_combined16 = models_training8 + models_testing8_b1

models_combined16_v2 = models_testing8_b1 + models_testing8_b2

models_combined24 = models_training8 + models_testing8_b1 + models_testing8_b2

models_combined25 = models_combined24 + ['face7']

models_test25_v2 = ['MB30798', '31_african_drums', 'foreign_cat', 'MB27585', 'MB31405',
'bullfrog', 'Colored_shirt_03M', 'MB30850',  'pear_obj_2',  '04_piano',
'lo_poly_animal_HRS_ARBN',  'MB31188',  'Doctor_pose02',  'lo_poly_animal_ANT_RED',  'MB27780',
'lo_poly_animal_TRTL_B', 'Nurse_pose01',  'household_aid_29',  'MB29346',  'laptop01',
'womens_Skirt_02M',  'womens_halterneck_06',  'MB29822', 'lo_poly_animal_DUCK',  'trousers_03'] # not used on monkeys for behavior

models_remaining14 = ['MB31620', 'interior_details_033_2', 'single_pineapple', 'pumpkin_3', 'interior_details_047_1',
 	'MB31015', 'fast_food_23_1', 'MB30071', 'leaves16', 'interior_details_103_2', 'jewelry_29', 'lo_poly_animal_TIGER_B',
 'lo_poly_animal_RB_TROUT', 'Tie_06'] # not used on monkeys for behavior or phys

models_objectome64_ordered = models_combined25 + models_test25_v2 + models_remaining14

models_MURI8 = ['single_banana', 'MB30102', 'kitchen_equipment_glassbowl', 'Floor_Lamp', 'MB27831',
 	 'kitchen_equipment_spatula', 'Professor_pose11', 'nails'] # addition objects for MURI project

models_combined72 = models_objectome64_ordered + models_MURI8

models_combined24_notxt = ['flarenut_spanner_tf', 'lo_poly_animal_RHINO_2_tf', 'womens_stockings_01M_tf', 'dromedary_tf',
 'MB30758_tf', 'lo_poly_animal_CHICKDEE_tf', 'womens_shorts_01M_tf', 'lo_poly_animal_ELE_AS1_tf',
 'build51_tf', 'Hanger_02_tf', 'interior_details_130_2_tf', 'MB28699_tf',
 'zebra_tf', 'interior_details_103_4_tf', '22_acoustic_guitar_tf', 'lo_poly_animal_BEAR_BLK_tf',
 'MB30203_tf', 'weimaraner_tf', 'lo_poly_animal_TRANTULA_tf', 'MB29874_tf',
 'antique_furniture_item_18_tf', 'calc01_tf', 'MB27346_tf', 'kitchen_equipment_knife2_tf']
	
models_simulation = ['face7'] + models_combined16

# screen for object sets based on machine objectome
models_screentest = []
models_screentest.append([ 'MB29874', 'weimaraner', 'interior_details_033_2', '31_african_drums', 'antique_furniture_item_18', 'bullfrog', 'kitchen_equipment_knife2', 'jewelry_29'])
models_screentest.append(['Tie_06', 'leaves16', 'kitchen_equipment_knife2', 'foreign_cat', 'womens_halterneck_06', 'womens_Skirt_02M', 'MB29346', 'MB29822'])
models_screentest.append(['MB29822', 'MB31015', 'interior_details_033_2', 'lo_poly_animal_TRANTULA', 'lo_poly_animal_TRTL_B', 'jewelry_29', 'MB27780', 'lo_poly_animal_DUCK'])
models_screentest.append(['MB29346', 'face7', 'lo_poly_animal_ANT_RED', 'lo_poly_animal_DUCK', 'fast_food_23_1', 'jewelry_29', 'weimaraner', '04_piano'])
models_screentest.append(['foreign_cat', 'MB27585', 'MB29346', 'pumpkin_3', 'bullfrog', 'single_pineapple', 'womens_halterneck_06', 'lo_poly_animal_DUCK'])

model_pairs_1 = [['flarenut_spanner','kitchen_equipment_knife2'],['lo_poly_animal_RHINO_2', 'lo_poly_animal_ELE_AS1'], ['lo_poly_animal_RHINO_2', 'zebra'], 
				['dromedary', 'weimaraner'], ['MB30758', 'MB29874'], ['womens_shorts_01M', 'face7'], ['build51', 'MB30203'], ['Hanger_02', 'interior_details_103_4'],
				['22_acoustic_guitar', 'kitchen_equipment_knife2'], ['antique_furniture_item_18', 'calc01']] # subsampled tasks for muscimol experiments

HVM_models = ['Apple_Fruit_obj', 'Apricot_obj', 'BAHRAIN', 'Beetle', 'CGTG_L', 'DTUG_L', 'ELEPHANT_M', 'GORILLA', 'LIONESS', 
    'MQUEEN_L', 'Peach_obj', 'Pear_obj', 'SISTER_L', 'Strawberry_obj', 'TURTLE_L', '_001', '_004', '_008', '_010', '_011', '_014',
    '_01_Airliner_2jetEngines', '_031', '_033', '_05_future', '_08', '_10', '_11', '_12', '_18', '_19_flyingBoat', '_37', '_38',
    '_44', 'alfa155', 'astra', 'bear', 'blCow', 'bmw325', 'bora_a', 'breed_pug', 'celica', 'clio', 'cruiser', 'f16', 'face0001',
    'face0002', 'face0003', 'face0004', 'face0005', 'face0006', 'face0007', 'face0008', 'hedgehog', 'junkers88', 'mig29',
     'motoryacht', 'raspberry_obj', 'rdbarren', 'sopwith', 'support', 'walnut_obj', 'watermelon_obj', 'z3']

HVM_10 = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',  'f16', '_001'] # first half of MURI20 image set, intersects with hvm set
#bear, elephant, table, face, 
#HVM new (6): car, dog, turtle, apple, chair, plane 

SYMBOLS_alpharoman = ['Letter_A', 'Letter_B', 'Letter_C', 'Letter_D', 'Letter_E','Letter_F', 'Letter_G', 'Letter_H', 'Letter_I', 'Letter_J',
    'Letter_K', 'Letter_L', 'Letter_M', 'Letter_N', 'Letter_O', 'Letter_P', 'Letter_Q', 'Letter_R', 'Letter_S', 'Letter_T',
    'Letter_U', 'Letter_V', 'Letter_W', 'Letter_X', 'Letter_Y', 'Letter_Z']
SYMBOLS_alphaarab = ['1_alif', '2_baa', '3_taa', '4_thaa', '5_jiim', '6_haa', '7_kha', '8_daal', '9_thaal', '10_raa', '11_zaay', '12_siin',
'13_shiin','14_saad', '15_daad', '16_taa', '17_thaa', '18_ayn', '19_ghayn', '20_faa', '21_qaaf', '22_kaaf', '23_laam', '24_miim', '25_nuun', 
'26_ha', '27_waaw', '28_yaa']
SYMBOLS_numbersarab = ['zero', 'one', 'two', 'three', 'four','five', 'six', 'seven', 'eight', 'nine']

SYMBOLS_all = SYMBOLS_alpharoman + SYMBOLS_alphaarab + SYMBOLS_numbersarab

""" Load metadata for different imagesets """
def hvm_meta():
    return pk.load(open('/mindhive/dicarlolab/u/rishir/stimuli/hvm/metadata.pkl', 'r'))

def hvm_stimpath():
    return '/mindhive/dicarlolab/u/rishir/stimuli/hvm/'
    
def objectome_meta():
    return pk.load(open('/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/metadata.pkl', 'r'))

def objectome_stimpath():
    return '/mindhive/dicarlolab/u/rishir/stimuli/objectome64s100/'


""" General utilities """

def concatenate_dictionary(dict_tuple, fns=None):
    concat_dict = {}
    if fns == None:
        fns = dict_tuple[0].keys()
    for fn in fns:
        concat_dict[fn] = []
        for dict_ in dict_tuple:
            concat_dict[fn] = concat_dict[fn] + list(dict_[fn])
    return concat_dict


""" Utilities relating to objectome objects"""

CANONICAL_ANGLES_obj24_old = {
    ('flarenut_spanner'):           {'rxy':0,   'rxz':0,   'ryz':90},
    ('lo_poly_animal_RHINO_2',
     'womens_stockings_01M',
     'lo_poly_animal_CHICKDEE',
     'lo_poly_animal_ELE_AS1',
     'build51',
     'lo_poly_animal_TRANTULA'):    {'rxy':0,   'rxz':90,   'ryz':270},
    ('22_acoustic_guitar', 
     'lo_poly_animal_BEAR_BLK', 
     'antique_furniture_item_18'):  {'rxy':0,   'rxz':90,   'ryz':180},
    ('MB30203'):                    {'rxy':0,   'rxz':90,   'ryz':90},
    ('dromedary',
     'MB30758', 
     'zebra', 
     'weimaraner', 
     'MB29874', 
     'MB27346'):                    {'rxy':90,  'rxz':0,    'ryz':180},
    ('womens_shorts_01M',
     'Hanger_02'):                  {'rxy':0,   'rxz':90,   'ryz':180},
    ('interior_details_103_4', 
     'kitchen_equipment_knife2'):   {'rxy':0,   'rxz':180,  'ryz':0},
    ('interior_details_130_2'):     {'rxy':180, 'rxz':0,    'ryz':270},
    ('calc01'):                     {'rxy':0,   'rxz':0,    'ryz':0},
    ('MB28699'):                    {'rxy':180, 'rxz':270,  'ryz':270},
}


CANONICAL_ANGLES_obj24 = {
    '22_acoustic_guitar': {'rxy':0, 'rxz':270, 'ryz': 0},
    'antique_furniture_item_18': {'rxy':0, 'rxz':270, 'ryz':0},
    'build51': {'rxy':0, 'rxz':270, 'ryz':90},
    'calc01': {'rxy':0,'rxz':0,'ryz':180},
    'dromedary': {'rxy':90, 'rxz':0,'ryz':0},
    'flarenut_spanner': {'rxy':0, 'rxz':180, 'ryz': 90},
    'interior_details_103_4': {'rxy':0, 'rxz':0, 'ryz': 0},
    'interior_details_130_2': {'rxy':0, 'rxz':0,'ryz':90},
    'lo_poly_animal_CHICKDEE': {'rxy':0, 'rxz':270, 'ryz':270},
    'lo_poly_animal_RHINO_2': {'rxy':0,'rxz':270, 'ryz':270},
    'MB28699': {'rxy':0, 'rxz':90, 'ryz':270},
    'MB29874' : {'rxy':90, 'rxz':0, 'ryz':0},
    'MB30203': {'rxy':0, 'rxz':90, 'ryz':270},
    'womens_shorts_01M': {'rxy':0, 'rxz':270, 'ryz':180},
    'zebra': {'rxy':90, 'rxz':0, 'ryz':0},
    'Hanger_02': {'rxy':0, 'rxz':270,'ryz':180},
    'kitchen_equipment_knife2': {'rxy':0, 'rxz':180, 'ryz':180},
    'lo_poly_animal_BEAR_BLK' : {'rxy':0, 'rxz':270, 'ryz':0},
    'womens_stockings_01M': {'rxy':0, 'rxz':270, 'ryz':90},
    'MB30758': {'rxy':90,  'rxz':180,    'ryz':0},
    'lo_poly_animal_ELE_AS1': {'rxy':0,   'rxz':270,   'ryz':270},
    'weimaraner':   {'rxy':90,  'rxz':180,    'ryz':180},
    'lo_poly_animal_TRANTULA': {'rxy':0,   'rxz':270,   'ryz':90},
    'MB27346': {'rxy':90,  'rxz':0,    'ryz':0},
}

def get_canonical_angle(objn):
    canonkeys = CANONICAL_ANGLES_obj24.keys()
    angles = [CANONICAL_ANGLES_obj24[cobj] for cobj in canonkeys if objn in cobj]
    if len(angles) == 0:
        return {'rxy':0, 'ryz':0, 'rxz':0}
    else:
        return angles[0]


import cPickle as pk
import numpy as np

MWKIMGINDEX_PATH = '/mindhive/dicarlolab/u/rishir/Manto/Images/obj24s100/'
obj24images = list(pk.load(open(MWKIMGINDEX_PATH + 'images24.pkl' ,'r')))


MWKIMGINDEX_PATH_HVM = '/mindhive/dicarlolab/u/rishir/stimuli_mwk/hvm10_test/'
hvm10images = pk.load(open(MWKIMGINDEX_PATH_HVM + 'images_dict.pkl' ,'r'))

def get_obj_indices(models):
	inds = []
	for m in models:
		inds.append(models_objectome64.index(m))
	return inds
	
def strip_objectomeFileName(fn):
    if len(fn) == 1:
        fn = fn[0]
        fn = fn.replace('_tf', '')
    return str(fn.split('/')[-1].split('.')[0]).replace('_label', '').replace('.png', '')

def get_index_MWorksRelative(img_id):
    if img_id in obj24images:
        return obj24images.index(img_id)+1
    elif img_id in hvm10images.keys():
        return hvm10images[img_id]+1
    else:
        return np.nan

def get_index_MWorksRelative_HVM(img_id):
    return hvm10images[img_id]

def get_hvm_3dmodelnames(models_hvm):
    HVM_models_v2 = []
    for mobj in models_hvm:
        if mobj[0] == '_':
            objout = mobj[1:] + '_POV_geom'
        elif mobj[:4] == 'face':
            objout = mobj
        else:
            objout = mobj + '_POV_geom'
        HVM_models_v2.append(objout)
    return HVM_models_v2