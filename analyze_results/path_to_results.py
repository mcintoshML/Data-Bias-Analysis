from constants import MODEL_DIR


def get_path(src='chexpert_80_20',dst='mimic',seed=20,shuffled=False):
    # --------- For custom re-runs - Use this path -----------------
    # shuf = 'shuffled_' if shuffled else ''
    # path = '%s%s_train_%s_valid_%sseed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)
    # return path

    
    shuf = 'shuffled_' if shuffled else ''
    path = '%s%s_train_%s_valid_%sseed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    if 'chexpert' in src or 'mimic' in src or 'combined' in src or 'nih' in src:
        path = '%s%s_train_%s_valid_%sN10_CP4800_seed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    if src in ('ptbxl','ludb') or dst in ('ptbxl','ludb'):
        path = '%s%s_train_%s_valid_%sseed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    elif 'ild' in path:
        path = '%s%s_train_%s_valid_%sN10_CP4800_eval2_seed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    elif 'icbhi' in path or 'just' in path:
        path = '%s%s_train_%s_valid_%sN50_CP4800_eval2_seed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    elif 'covid' in path:
        if shuffled:
            path = '%s%s_train_%s_valid_%sN50_CP4800_eval4_seed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)
        else:
            path = '%s%s_train_%s_valid_%sN10_CP4800_eval4_seed%d/res_ops/%s_valid.npz'%(MODEL_DIR,src,src,shuf,seed,dst)

    return path