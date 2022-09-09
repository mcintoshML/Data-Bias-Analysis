import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interpolate
from joblib import Parallel,delayed
from tqdm import tqdm
from io import StringIO
import scipy.stats
import scipy

from analyze_results.path_to_results import get_path
from constants import MODEL_DIR
from engine import utils
import constants


classes_to_evaluate = {
    'mimic': constants.CXRAY_CL,
    'chexpert_80_20': constants.CXRAY_CL,
    'nih': constants.CXRAY_CL,
    'combined': constants.CXRAY_CL,
    'combined_nihmimic': constants.CXRAY_CL,
    'covid_kaggle':constants.COVID_DIS_CL,
    'covid_internal':constants.COVID_DIS_CL,
    'ild':constants.ILD_DIS_CL,
    'icbhi':constants.LUNG_DIS_CL,
    'just':constants.LUNG_DIS_CL,
    'ptbxl':constants.LUDB_DIS_CL,
    'ludb':constants.LUDB_DIS_CL,
}

graph_mapping = {
        'ild_plan':'ILD Plan','ild_TR':'ILD Diag',
        'chexpert':'CXP',
        'chexpert_80_20':'CXP','mimic':'MIMIC','nih':'NIH',
        'combined':'CXP+MIMIC',
        'combined_nihmimic':'NIH+MIMIC',
        'covid_kaggle':'COVID-Kaggle','covid_internal':'COVID-Internal',
        'covid_internal_port':'COVID-Internal-Portable','covid_internal_chest':'COVID-Internal-Regular',
        'icbhi':'ICBHI','just':'JUST',
        'ludb':'LUDB','ptbxl':'PTBXL',
    }


cache = {} #Speed up bootstrapping
ALL_FPR = np.linspace(0,1,10**3)

def get_data(path):
    if path not in cache:
        data = np.load(path)
        df1 = pd.DataFrame(data['y_true'])
        df2 = pd.DataFrame(data['y_scores'])
        df = pd.concat([df2,df1],axis=1)
        cache[path] = df
    return cache[path]

def standardize_auc_curves(fpr,tpr):
    fpr = np.concatenate([[0],fpr])
    tpr = np.concatenate([[0],tpr])
    all_tpr = interpolate.interp1d(fpr,tpr,kind='nearest')(ALL_FPR)
    return ALL_FPR,all_tpr

def get_res(src='chexpert_80_20',dst='mimic',seed=20,shuffled=False,bootstrap_seed=None):
    path = get_path(src=src,dst=dst,seed=seed,shuffled=shuffled)
    df = get_data(path)

    if bootstrap_seed:
        df = df.sample(frac=1,replace=True,random_state=bootstrap_seed)
    
    half_idx = df.shape[1]//2
    y_true = df.iloc[:,half_idx:].values
    y_scores = df.iloc[:,:half_idx].values
    y_scores = utils.sigmoid(y_scores)

    fpr = {}
    tpr = {}
    roc_auc = {}

    eval_classes = classes_to_evaluate[src]

    # If any labels dont have positive samples, AUC cannot be computed
    # Ignore this bootstrapped result by returning None
    if np.any(np.sum(y_true[:,eval_classes],axis=0)==0):
        print(np.sum(y_true[:,eval_classes],axis=0))
        return None

    for cl in eval_classes:
        fpr[cl], tpr[cl], _ = metrics.roc_curve(y_true[:, cl], y_scores[:, cl],drop_intermediate=True)     
        roc_auc[cl] = metrics.auc(fpr[cl], tpr[cl])


    # Calculate average AUROC across the classes 
    all_fpr = np.unique(np.concatenate([fpr[i] for i in eval_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in eval_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(eval_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    cl = 'macro'
    res = {'fpr':fpr[cl],'tpr':tpr[cl],'roc_auc':roc_auc[cl]}
    res['fpr'],res['tpr'] = standardize_auc_curves(res['fpr'],res['tpr'])
    return res

def show_details(res):
    print(res['tpr'][:5],res['fpr'][:5],res['tpr'].shape)

def rotate(x,y,theta=45):
    # rotate vectors by theta degrees
    X = np.column_stack([x,y])
    theta = np.pi * (theta/180)
    m = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    X_rot = np.matmul(X,m)
    x,y = X_rot[:,0],X_rot[:,1]
    return x,y    

def subtract_along_y_eq_minus_x(x1,y1,x2,y2):
    #rotate 45 degrees
    x1_45,y1_45 = rotate(x1,y1,theta=45)
    x2_45,y2_45 = rotate(x2,y2,theta=45)

    # align x axis
    x3_45 = np.unique(np.concatenate([x1_45,x2_45]))
    y1_45 = np.interp(x3_45,x1_45,y1_45)
    y2_45 = np.interp(x3_45,x2_45,y2_45)
    
    # subtract y
    y3_45 = y1_45 - y2_45

    # rotate back
    x3,y3 = rotate(x3_45,y3_45,theta=-45)

    # standardize x axis
    y3 = np.interp(ALL_FPR,x3,y3)
    x3 = ALL_FPR

    return x3,y3


def get_dab_projection(src='chexpert8020',dst='mimic',seed=20,bootstrap_seed=None):

    res_src = get_res(src=src,dst=src,seed=seed,shuffled=False,bootstrap_seed=bootstrap_seed)
    res_src_shuff = get_res(src=src,dst=src,seed=seed,shuffled=True,bootstrap_seed=bootstrap_seed)
    res_dst = get_res(src=src,dst=dst,seed=seed,shuffled=False,bootstrap_seed=bootstrap_seed)

    if res_src is None or res_src_shuff is None or res_dst is None:
        return None

    res_proj = {}
    res_proj['fpr'],res_proj['tpr'] = subtract_along_y_eq_minus_x(x1=res_src['fpr'],y1=res_src['tpr'],x2=res_src_shuff['fpr'],y2=res_src_shuff['tpr'])
    res_proj['roc_auc'] = res_src['roc_auc'] - res_src_shuff['roc_auc'] + .5

    res = np.concatenate([res_src['tpr'],res_src_shuff['tpr'],res_dst['tpr'],res_proj['tpr'],
        [res_src['roc_auc']],[res_src_shuff['roc_auc']],[res_dst['roc_auc']],[res_proj['roc_auc']]])

    res = np.expand_dims(res,0)
    return res

def mean_confidence_interval(data, confidence=0.95):
    m = np.mean(data)
    intervals = np.percentile(data,[100*(1-confidence)/2,100*(1-(1-confidence)/2)])
    diff1 = m - intervals[0]
    diff2 = intervals[1] - m
    return m,diff1,diff2

def plot_graph(src='chexpert8020',dst='mimic',seed=20,n_bootstraps=1000,graph_dir='analyze_results/auc_graphs/'):

    # Get projection across n_bootstraps
    res = Parallel(n_jobs=1)(delayed(get_dab_projection)(src=src,dst=dst,seed=seed,bootstrap_seed=idx) for idx in tqdm(np.arange(n_bootstraps),leave=True))
    res = np.concatenate([x for x in res if x is not None])

    # Caclulate mean and confidence intervals across the bootstraps
    mean_ci1_ci2 = np.array([mean_confidence_interval(res[:,idx],confidence=0.95) for idx in range(res.shape[-1])])
    mean,ci1,ci2 = mean_ci1_ci2[:,0],mean_ci1_ci2[:,1],mean_ci1_ci2[:,2]

    N = len(ALL_FPR)
    src_tpr_mean,src_shuff_tpr_mean,dst_tpr_mean,proj_tpr_mean = np.split(mean[:4*N],4)
    src_auc,src_shuff_auc,dst_auc,proj_auc = np.split(mean[4*N:],4)
    src_tpr_ci1,src_shuff_tpr_ci1,dst_tpr_ci1,proj_tpr_ci1 = np.split(ci1[:4*N],4)
    src_auc_ci1,src_shuff_auc_ci1,dst_auc_ci1,proj_auc_ci1 = np.split(ci1[4*N:],4)
    src_tpr_ci2,src_shuff_tpr_ci2,dst_tpr_ci2,proj_tpr_ci2 = np.split(ci2[:4*N],4)
    src_auc_ci2,src_shuff_auc_ci2,dst_auc_ci2,proj_auc_ci2 = np.split(ci2[4*N:],4)

    # Plotting the graphs
    mapping = graph_mapping

    src = mapping[src]
    dst = mapping[dst]

    plt.plot(ALL_FPR,src_tpr_mean,label='%s AUC : %.2f (%.2f,%.2f)'%(src,src_auc,src_auc-src_auc_ci1,src_auc+src_auc_ci2),color='red')
    plt.fill_between(ALL_FPR,src_tpr_mean+src_tpr_ci2,src_tpr_mean-src_tpr_ci1,color='red',alpha=.4)
    plt.plot(ALL_FPR,src_shuff_tpr_mean,label='%s Shuffled AUC : %.2f (%.2f,%.2f)'%(src,src_shuff_auc,src_shuff_auc-src_shuff_auc_ci1,src_shuff_auc+src_shuff_auc_ci2),color='grey')
    plt.fill_between(ALL_FPR,src_shuff_tpr_mean+src_shuff_tpr_ci2,src_shuff_tpr_mean-src_shuff_tpr_ci1,color='grey',alpha=.4)
    plt.plot([0,1],[0,1],linestyle='--',color='black')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    path = utils.get_prepared_path('%s%s_%s_%d_1.png'%(graph_dir,src,dst,seed))
    plt.legend(loc='lower right')
    plt.savefig(path,dpi=450,bbox_inches='tight')
    plt.close()
    print(path)

    plt.plot(ALL_FPR,dst_tpr_mean,label='%s AUC : %.2f (%.2f,%.2f)'%(dst,dst_auc,dst_auc-dst_auc_ci1,dst_auc+dst_auc_ci2),color='green')
    plt.fill_between(ALL_FPR,dst_tpr_mean+dst_tpr_ci2,dst_tpr_mean-dst_tpr_ci1,color='green',alpha=.4)
    plt.plot(ALL_FPR,proj_tpr_mean,label='%s Projection AUC : %.2f (%.2f,%.2f)'%(dst,proj_auc,proj_auc-proj_auc_ci1,proj_auc+proj_auc_ci2),color='orange')
    plt.fill_between(ALL_FPR,proj_tpr_mean+proj_tpr_ci2,proj_tpr_mean-proj_tpr_ci1,color='orange',alpha=.4)
    plt.plot([0,1],[0,1],linestyle='--',color='black')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    path = utils.get_prepared_path('%s%s_%s_%d_2.png'%(graph_dir,src,dst,seed))
    plt.legend(loc='lower right')
    plt.savefig(path,dpi=450,bbox_inches='tight')
    plt.close()
    print(path)


def main():


    # plot_graph(src='icbhi',dst='just',seed=20,n_bootstraps=10)
    plot_graph(src='ptbxl',dst='ludb',seed=20,n_bootstraps=1000)


    # plot_graph(src='chexpert_80_20',dst='mimic',seed=20,n_bootstraps=1000)
    # plot_graph(src='chexpert_80_20',dst='nih',seed=20,n_bootstraps=1000)
    # plot_graph(src='mimic',dst='chexpert_80_20',seed=20,n_bootstraps=1000)
    # plot_graph(src='mimic',dst='nih',seed=20,n_bootstraps=1000)
    # plot_graph(src='combined',dst='nih',seed=20,n_bootstraps=1000)
    # plot_graph(src='combined_nihmimic',dst='chexpert_80_20',seed=20,n_bootstraps=1000)
    # plot_graph(src='covid_kaggle',dst='covid_internal',seed=30,n_bootstraps=1000)
    # plot_graph(src='ild_TR',dst='ild_plan',seed=20,n_bootstraps=1000)
    # plot_graph(src='icbhi',dst='just',seed=20,n_bootstraps=1000)
    # plot_graph(src='ptbxl',dst='ludb',seed=20,n_bootstraps=1000)
    # plot_graph(src='covid_kaggle',dst='covid_internal_port',seed=30,n_bootstraps=1000)
    # plot_graph(src='covid_kaggle',dst='covid_internal_chest',seed=30,n_bootstraps=1000)


