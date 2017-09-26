all_models = [
'GOOGLENETv3_seed0mdl_mixed_35x35x288a_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x768c_pca',
'GOOGLENETv3_seed1mdl_aux_logits_pca',
'GOOGLENETv3_seed0mdl_mixed_35x35x288b_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x768b_pca',
'GOOGLENETv3_seed1mdl_logits_pca',
'GOOGLENETv3_seed0mdl_flattened_last_mixed_pca',
'GOOGLENETv3_seed0mdl_mixed_8x8x2048b_pca',
'GOOGLENETv3_seed2mdl_mixed_8x8x2048a_pca',
'GOOGLENETv3_seed1mdl_mixed_17x17x1280a_pca',
'GOOGLENETv3_seed0mdl_aux_logits_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x768e_pca',
'GOOGLENETv3_seed2mdl_logits_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x768a_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x1280a_pca',
'GOOGLENETv3_seed1mdl_flattened_last_mixed_pca',
'GOOGLENETv3_seed2mdl_mixed_8x8x2048b_pca',
'GOOGLENETv3_seed1mdl_mixed_35x35x288b_pca',
'GOOGLENETv3_seed2mdl_mixed_35x35x256a_pca',
'GOOGLENETv3_seed2mdl_mixed_35x35x288b_pca',
'GOOGLENETv3_seed2mdl_flattened_last_mixed_pca',
'GOOGLENETv3_seed0mdl_mixed_35x35x256a_pca',
'GOOGLENETv3_seed0mdl_logits_pca',
'GOOGLENETv3_seed0mdl_mixed_8x8x2048a_pca',
'GOOGLENETv3_seed2mdl_mixed_35x35x288a_pca',
'GOOGLENETv3_seed1mdl_mixed_8x8x2048a_pca',
'GOOGLENETv3_seed0mdl_mixed_17x17x768d_pca',
'GOOGLENETv3_seed1mdl_mixed_17x17x768a_pca',
'GOOGLENETv3_seed2mdl_mixed_17x17x1280a_pca',
'GOOGLENETv3_seed2mdl_mixed_17x17x768b_pca',
'GOOGLENETv3_seed1mdl_mixed_17x17x768d_pca',
'GOOGLENETv3_seed1mdl_mixed_35x35x256a_pca',
'GOOGLENETv3_seed2mdl_mixed_17x17x768d_pca',
'GOOGLENETv3_seed2mdl_mixed_17x17x768e_pca',
'GOOGLENETv3_seed1mdl_mixed_35x35x288a_pca',
'GOOGLENETv3_seed2mdl_mixed_17x17x768a_pca',
'GOOGLENETv3_seed1mdl_mixed_17x17x768b_pca',
'GOOGLENETv3_seed1mdl_mixed_8x8x2048b_pca',
'GOOGLENETv3_seed1mdl_mixed_17x17x768c_pca',
'GOOGLENETv3_finetune60000_seed1_flattened_last_mixed',
'GOOGLENETv3_finetune60000_seed3_flattened_last_mixed',
'GOOGLENETv3_finetune60000_seed0_flattened_last_mixed',
'GOOGLENETv3_finetune60000_seed4_flattened_last_mixed',
'GOOGLENETv3_finetune60000_seed2_flattened_last_mixed',
'GOOGLENETv3_seed01234mdl_flattened_last_mixed',
'GOOGLENETv3_seed1234mdl_flattened_last_mixed',
'GOOGLENETv3_seed0234mdl_flattened_last_mixed',
'GOOGLENETv3_seed0134mdl_flattened_last_mixed',
'GOOGLENETv3_seed0124mdl_flattened_last_mixed',
'GOOGLENETv3_seed0123mdl_flattened_last_mixed',
'GOOGLENETv3_seed234mdl_flattened_last_mixed',
'GOOGLENETv3_seed134mdl_flattened_last_mixed',
'GOOGLENETv3_seed124mdl_flattened_last_mixed',
'GOOGLENETv3_seed123mdl_flattened_last_mixed',
'GOOGLENETv3_seed034mdl_flattened_last_mixed',
'GOOGLENETv3_seed024mdl_flattened_last_mixed',
'GOOGLENETv3_seed023mdl_flattened_last_mixed',
'GOOGLENETv3_seed014mdl_flattened_last_mixed',
'GOOGLENETv3_seed013mdl_flattened_last_mixed',
'GOOGLENETv3_seed012mdl_flattened_last_mixed',
'GOOGLENETv3_seed34mdl_flattened_last_mixed',
'GOOGLENETv3_seed24mdl_flattened_last_mixed',
'GOOGLENETv3_seed23mdl_flattened_last_mixed',
'GOOGLENETv3_seed14mdl_flattened_last_mixed',
'GOOGLENETv3_seed13mdl_flattened_last_mixed',
'GOOGLENETv3_seed12mdl_flattened_last_mixed',
'GOOGLENETv3_seed04mdl_flattened_last_mixed',
'GOOGLENETv3_seed03mdl_flattened_last_mixed',
'GOOGLENETv3_seed02mdl_flattened_last_mixed',
'GOOGLENETv3_seed01mdl_flattened_last_mixed',
'GOOGLENETv3_seed0mdl_flattened_last_mixed',
'GOOGLENETv3_seed4mdl_flattened_last_mixed',
'GOOGLENETv3_seed2mdl_flattened_last_mixed',
'GOOGLENETv3_seed3mdl_flattened_last_mixed',
'GOOGLENETv3_seed1mdl_flattened_last_mixed',
'V1',
'HMAX',
'VGG_fc6',
'VGG_fc7',
'VGG_fc8',
'ALEXNET_fc6',
'ALEXNET_fc7',
'ALEXNET_fc8',
'ZEILER_fc6',
'GOOGLENET_pool5',
'RESNET101_conv5',
'GOOGLENETv3_mdl_flattened_last_mixed',
'GOOGLENETv3_retina_mdl_flattened_last_mixed',
'GOOGLENETv3_synth34000__mdl_flattened_last_mixed']

all_subjects = [
 'monk_SUBPOOL_Bento',
 'monk_SUBPOOL_Magneto',
 'monk_SUBPOOL_Manto',
 'monk_SUBPOOL_Nano',
 'monk_SUBPOOL_Pablo',
 'monk_SUBPOOL_BentoMagneto',
 'monk_SUBPOOL_BentoPablo',
 'monk_SUBPOOL_MagnetoManto',
 'monk_SUBPOOL_MagnetoNano',
 'monk_SUBPOOL_MagnetoPablo',
 'monk_SUBPOOL_MagnetoPicasso',
 'monk_SUBPOOL_MagnetoZico',
 'monk_SUBPOOL_MantoPablo',
 'monk_SUBPOOL_NanoPablo',
 'monk_SUBPOOL_PabloPicasso',
 'monk_SUBPOOL_PabloZico',
 'monk_SUBPOOL_BentoMantoPablo',
 'monk_SUBPOOL_BentoNanoPablo',
 'monk_SUBPOOL_BentoPabloZico',
 'monk_SUBPOOL_MagnetoNanoZico',
 'monk_SUBPOOL_MantoNanoPablo',
 'monk_SUBPOOL_MantoPabloZico',
 'monk_SUBPOOL_NanoPabloZico',
 'hum_SUBPOOL_AURAZWWGQBQKW',
 'monk_SUBPOOL_MantoNano',
 'hum_SUBPOOL_A3G2NE6QE5W5R',
 'hum_SUBPOOL_A3Z1W0ACQDGGC',
 'monk_SUBPOOL_BentoNano',
 'hum_SUBPOOL_A8SNALQ3K98RB',
 'hum_SUBPOOL_A2CXEAMWU2SFV3',
 'monk_SUBPOOL_Picasso',
 'monk_SUBPOOL_BentoManto',
 'monk_SUBPOOL_MantoZico',
 'monk_SUBPOOL_MantoPicasso',
 'monk_SUBPOOL_NanoZico',
 'monk_SUBPOOL_BentoPicasso',
 'monk_SUBPOOL_NanoPicasso',
 'monk_SUBPOOL_Zico',
 'monk_SUBPOOL_BentoZico',
 'monk_SUBPOOL_BentoMantoNano',
 'monk_SUBPOOL_BentoNanoZico',
 'monk_SUBPOOL_MantoNanoZico',
 'monk_SUBPOOL_PicassoZico',
 'monk_SUBPOOL_BentoMantoZico',
 'monk_SUBPOOL_NanoPicassoZico',
 'monk_SUBPOOL_BentoMagnetoManto',
 'monk_SUBPOOL_BentoMagnetoNano',
 'monk_SUBPOOL_BentoMagnetoPablo',
 'monk_SUBPOOL_BentoMagnetoPicasso',
 'monk_SUBPOOL_BentoMagnetoZico',
 'monk_SUBPOOL_BentoPabloPicasso',
 'monk_SUBPOOL_MagnetoMantoNano',
 'monk_SUBPOOL_MagnetoMantoPablo',
 'monk_SUBPOOL_MagnetoMantoPicasso',
 'monk_SUBPOOL_MagnetoMantoZico',
 'monk_SUBPOOL_MagnetoNanoPablo',
 'monk_SUBPOOL_MagnetoNanoPicasso',
 'monk_SUBPOOL_MagnetoPabloPicasso',
 'monk_SUBPOOL_MagnetoPabloZico',
 'monk_SUBPOOL_MagnetoPicassoZico',
 'monk_SUBPOOL_MantoPabloPicasso',
 'monk_SUBPOOL_NanoPabloPicasso',
 'monk_SUBPOOL_PabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNano',
 'monk_SUBPOOL_BentoMagnetoMantoPablo',
 'monk_SUBPOOL_BentoMagnetoMantoPicasso',
 'monk_SUBPOOL_BentoMagnetoMantoZico',
 'monk_SUBPOOL_BentoMagnetoNanoPablo',
 'monk_SUBPOOL_BentoMagnetoNanoPicasso',
 'monk_SUBPOOL_BentoMagnetoNanoZico',
 'monk_SUBPOOL_BentoMagnetoPabloPicasso',
 'monk_SUBPOOL_BentoMagnetoPabloZico',
 'monk_SUBPOOL_BentoMagnetoPicassoZico',
 'monk_SUBPOOL_BentoMantoNanoPablo',
 'monk_SUBPOOL_BentoMantoPabloPicasso',
 'monk_SUBPOOL_BentoMantoPabloZico',
 'monk_SUBPOOL_BentoNanoPabloPicasso',
 'monk_SUBPOOL_BentoNanoPabloZico',
 'monk_SUBPOOL_BentoPabloPicassoZico',
 'monk_SUBPOOL_MagnetoMantoNanoPablo',
 'monk_SUBPOOL_MagnetoMantoNanoPicasso',
 'monk_SUBPOOL_MagnetoMantoNanoZico',
 'monk_SUBPOOL_MagnetoMantoPabloPicasso',
 'hum_SUBPOOL_A3G2NE6QE5W5RAURAZWWGQBQKW',
 'hum_SUBPOOL_A8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A3Z1W0ACQDGGCAURAZWWGQBQKW',
 'hum_SUBPOOL_A3Z1W0ACQDGGCA8SNALQ3K98RB',
 'hum_SUBPOOL_A3G2NE6QE5W5RA8SNALQ3K98RB',
 'hum_SUBPOOL_A3G2NE6QE5W5RA3Z1W0ACQDGGC',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A8SNALQ3K98RB',
 'hum_SUBPOOL_A2CXEAMWU2SFV3AURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5R',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3Z1W0ACQDGGC',
 'monk_SUBPOOL_BentoNanoPicasso',
 'monk_SUBPOOL_BentoMantoPicasso',
 'monk_SUBPOOL_MantoNanoPicasso',
 'monk_SUBPOOL_BentoPicassoZico',
 'monk_SUBPOOL_MantoPicassoZico',
 'monk_SUBPOOL_MantoNanoPabloZico',
 'monk_SUBPOOL_MagnetoNanoPabloZico',
 'monk_SUBPOOL_MantoNanoPabloPicasso',
 'monk_SUBPOOL_MagnetoMantoPabloZico',
 'monk_SUBPOOL_MagnetoNanoPabloPicasso',
 'monk_SUBPOOL_MantoNanoPicassoZico',
 'monk_SUBPOOL_BentoMantoNanoPicasso',
 'monk_SUBPOOL_MantoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPicasso',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPablo',
 'monk_SUBPOOL_NanoPabloPicassoZico',
 'monk_SUBPOOL_BentoMantoNanoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoZico',
 'monk_SUBPOOL_BentoNanoPicassoZico',
 'monk_SUBPOOL_MagnetoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoNanoPabloZico',
 'monk_SUBPOOL_MagnetoNanoPicassoZico',
 'monk_SUBPOOL_MagnetoMantoPicassoZico',
 'monk_SUBPOOL_BentoMantoPicassoZico',
 'monk_SUBPOOL_BentoMagnetoNanoPabloPicasso',
 'monk_SUBPOOL_BentoMantoNanoPabloZico',
 'monk_SUBPOOL_BentoMantoNanoPabloPicasso',
 'monk_SUBPOOL_BentoMagnetoMantoPicassoZico',
 'monk_SUBPOOL_MagnetoMantoNanoPabloPicasso',
 'monk_SUBPOOL_BentoMagnetoMantoPabloPicasso',
 'hum_SUBPOOL_A3G2NE6QE5W5RA3Z1W0ACQDGGCA8SNALQ3K98RB',
 'hum_SUBPOOL_A3G2NE6QE5W5RA3Z1W0ACQDGGCAURAZWWGQBQKW',
 'hum_SUBPOOL_A3G2NE6QE5W5RA8SNALQ3K98RBAURAZWWGQBQKW',
 'monk_SUBPOOL_BentoMagnetoNanoPicassoZico',
 'monk_SUBPOOL_BentoMantoPabloPicassoZico',
 'monk_SUBPOOL_MagnetoMantoNanoPabloZico',
 'monk_SUBPOOL_BentoMagnetoMantoPabloZico',
 'monk_SUBPOOL_BentoMagnetoPabloPicassoZico',
 'monk_SUBPOOL_MagnetoMantoNanoPicassoZico',
 'monk_SUBPOOL_BentoNanoPabloPicassoZico',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3Z1W0ACQDGGCA8SNALQ3K98RB',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3Z1W0ACQDGGCAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA8SNALQ3K98RB',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA3Z1W0ACQDGGC',
 'hum_SUBPOOL_A3Z1W0ACQDGGCA8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA3Z1W0ACQDGGCAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3Z1W0ACQDGGCA8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A3G2NE6QE5W5RA3Z1W0ACQDGGCA8SNALQ3K98RBAURAZWWGQBQKW',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA3Z1W0ACQDGGCA8SNALQ3K98RB',
 'hum_SUBPOOL_A2CXEAMWU2SFV3A3G2NE6QE5W5RA3Z1W0ACQDGGCA8SNALQ3K98RBAURAZWWGQBQKW',
 'monk_SUBPOOL_BentoMantoNanoPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPabloPicasso',
 'monk_SUBPOOL_MagnetoMantoNanoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPabloZico',
 'monk_SUBPOOL_MagnetoMantoPabloPicassoZico',
 'monk_SUBPOOL_MagnetoNanoPabloPicassoZico',
 'monk_SUBPOOL_BentoMantoNanoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPabloPicassoZico',
 'monk_SUBPOOL_MantoNanoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoNanoPicassoZico',
 'monk_SUBPOOL_BentoMagnetoMantoPabloPicassoZico',
 'monk_SUBPOOL_BentoMagnetoNanoPabloPicassoZico']


def get_subject_list_meta(ignore_hum_subs=None, ignore_monk_subs=None):
    if ignore_hum_subs is None:
        ignore_hum_subs = ['AURAZWWGQBQKW']
    if ignore_monk_subs is None:
        ignore_monk_subs = ['Magneto', 'Pablo'] # too few trials for I2

    ignore_subs = ignore_monk_subs + ignore_hum_subs
    nchar_per_sub = 13
    hum_subpool_list_meta = []
    monk_subpool_list_meta = []

    for sub in all_subjects:
        if sum([sub.find(is_) for is_ in ignore_subs]) > 0:
            continue
        fname = sub.replace('_SUBPOOL_', '').replace('hum', '').replace('monk', '')
        if 'hum' in sub:
            nsubs = len(fname) / nchar_per_sub
            hum_subpool_list_meta.append([nsubs, sub])
        elif 'monk' in sub:
            nsubs = sum([s.isupper() for s in fname])
            monk_subpool_list_meta.append([nsubs, sub])

    return hum_subpool_list_meta, monk_subpool_list_meta
