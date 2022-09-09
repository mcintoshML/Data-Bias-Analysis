# Dont forget the / at the end of the paths

# unzip data to these folders
CHEXPERT_DIR = '/cluster/projects/mcintoshgroup/CheXpert-v1.0-small/'
MIMIC_DIR = '/cluster/projects/mcintoshgroup/MIMIC-CXR/MIMIC-CXR/physionet.org/files/mimic-cxr/2.0.0/files/'
NIH_DIR = '/cluster/projects/mcintoshgroup/NIH-14/'
CXRAY_SZ = 320
CXRAY_DIS = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity','Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis','Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture','Support Devices']
CXRAY_CL = [2,5,6,8,10]

COVID_KAGGLE_DIR = '/cluster/projects/mcintoshgroup/COVID-19_Radiography_Dataset/'
COVID_INT_DIR1 = '/cluster/projects/mcintoshgroup/COVID-PositiveOnly/'
COVID_INT_DIR2 = '/cluster/projects/mcintoshgroup/COVID-imaging/'
COVID_XRAY_SZ = 256
COVID_DIS = ['COVID','Normal','Lung_Opacity','Viral Pneumonia']
COVID_DIS_CL = [0,1]

JUST_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/AudioBias/Test_Data/Audio/'
ICBHI_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/AudioBias/ICBHI_final_database/'
PROCESSED_MEL_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/AudioLungSounds/mel_spec/'
MEL_SPEC_SZ = 224 
LUNG_DIS = ['Normal','Abnormal']
LUNG_DIS_CL = [0,1]

ILD_DIR = '/cluster/projects/mcintoshgroup/ILD/'
# ILD_DIS = ['Normal','Abnormal'] - # Changing since labels were originally flipped
ILD_DIS = ['Abnormal','Normal'] 
ILD_DIS_CL = [0,1]

PTBXL_DIR = '/cluster/projects/mcintoshgroup/Results/PTB-XL/'
PTBXL_DIS = ['CD','HYP','MI','STTC','NORM']
PTBXL_DIS_CL = [0,1,2,3,4]

LUDB_DIR = '/cluster/projects/mcintoshgroup/Results/LUDB/'
LUDB_DIS = ['CD','HYP']
LUDB_DIS_CL = [0,1]

# Save models here
BASE_DATA_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/'
MODEL_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/models_v1/'
# MODEL_DIR = '/cluster/projects/mcintoshgroup/BiasAnalysis/models/'


MEAN_STD = {

    'chexpert_train':[[128.2121,],[74.0974,]],
    'mimic_train':[[2477.1777,],[1103.6028,]],
    'nih_train':[[128.9558,],[59.0587,]],
    'covid_kaggle_train':[[129.5715,],[58.9904,]],
    # OLD # 'covid_internal_train':[[2330.8491,],[776.7404,]],
    'covid_internal_train':[[2347.5303,],[759.0811,]],

    'ild_TR_train':[-584.2516,492.6476],
    'ild_plan_train':[-749.2117,415.7507],
    
    # OLD
    # 'just_train': [[20.3834,  8.4188, 24.1917],[43.0854, 17.9024, 36.8831]],
    # 'icbhi_train': [[54.8954, 20.1839, 61.6515],[55.9522, 24.3787, 39.3875]],  

    'just_train': [[24.7278,  9.9977, 27.5706],[50.5934, 22.3349, 39.7978]],
    'icbhi_train': [[62.2622, 22.9156, 68.2407],[61.6764, 28.8834, 38.7871]],

    # 12 Channel
    'ptbxl_train':[
        [-0.0020, -0.0015,  0.0006,  0.0018, -0.0012, -0.0004,  0.0002, -0.0009,-0.0016, -0.0014, -0.0008, -0.0024],
        [0.1412, 0.1469, 0.1320, 0.1283, 0.1151, 0.1200, 0.1903, 0.2886, 0.2851,0.2653, 0.2386, 0.2012]
        ],

    'ludb_train':[
        [0.0005, 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0003, 0.0002,0.0002, 0.0002, 0.0003],
        [0.1167, 0.1142, 0.1054, 0.1187, 0.1065, 0.1097, 0.1198, 0.1180, 0.1162,0.1108, 0.1142, 0.1183]
    ]


}

DIS_CLASSES = {
    'chexpert_80_20': CXRAY_DIS,
    'mimic': CXRAY_DIS,
    'combined': CXRAY_DIS,
    'nih': CXRAY_DIS,
    'combined_nihmimic': CXRAY_DIS,
    
    'covid_kaggle': COVID_DIS,
    'covid_internal': COVID_DIS,
}