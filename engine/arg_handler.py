import argparse
import copy


mimic = {
    'name':'mimic',
    'exp':'mimic',
    'seed':20,

    'arch':'dn121',
    'num_classes':14,
    'pretrained':False,
    'num_channels':1,
    
    'lr':1e-4,
    'betas':(0.9, 0.999),
    'num_epochs':10,
    'tensor_log':False,
    'chkpt_every':4800,
    'early_stop':10,

    'num_samples':None,
    'preload_to_ram':False,
    'shuffle_pixels': False, 

    'train_on':'mimic_train',
    'valid_on':'mimic_valid',
    'test_on':'chexpert_80_20_valid',
    
    'task':'eval',
    'disable_tqdm':False,
    'ngpu':2,

    'eval_classes':[2, 5, 6, 8, 10],
    'test_eval_classes':'same',
    'batch_size':16,
    'num_workers':8,
    'img_sz':320,

    'ds_class':'rad',
    'cuda_device':'cuda:0',
    'folds':None,
}

mimic = argparse.Namespace(**mimic)

chexpert_80_20 = copy.deepcopy(mimic)
chexpert_80_20.train_on = 'chexpert_80_20_train'
chexpert_80_20.valid_on = 'chexpert_80_20_valid'
chexpert_80_20.test_on = 'chexpert_80_20_valid'

combined = copy.deepcopy(mimic)
combined.train_on = 'combined_train'
combined.valid_on = 'combined_valid'
combined.test_on = 'combined_valid'

combined_nihmimic = copy.deepcopy(mimic)
combined_nihmimic.train_on = 'combined_nihmimic_train'
combined_nihmimic.valid_on = 'combined_nihmimic_valid'
combined_nihmimic.test_on = 'combined_nihmimic_valid'

covid_kaggle = copy.deepcopy(mimic)
covid_kaggle.num_classes = 4
covid_kaggle.eval_classes = [0,1,2,3]
covid_kaggle.test_eval_classes = [0,1]
covid_kaggle.img_sz = 256
covid_kaggle.train_on = 'covid_kaggle_train'
covid_kaggle.valid_on = 'covid_kaggle_valid'
covid_kaggle.test_on = 'covid_kaggle_valid'

ild3d = copy.deepcopy(mimic)
ild3d.num_classes = 2
ild3d.eval_classes = [0,1]
ild3d.train_on = 'ild_TR_train'
ild3d.valid_on = 'ild_TR_valid'
ild3d.test_on = 'ild_plan_train'
ild3d.arch = 'vgg3d'
ild3d.ds_class = 'ild'


icbhi = copy.deepcopy(mimic)
icbhi.num_classes = 2
icbhi.eval_classes = [0,1]
icbhi.img_sz = 224
icbhi.arch = 'resnet34'
icbhi.train_on = 'icbhi_train'
icbhi.valid_on = 'icbhi_valid'
icbhi.test_on = 'just_all'
icbhi.num_epochs = 50
icbhi.early_stop = 10
icbhi.batch_size = 64
icbhi.ngpu = 1
icbhi.preload_to_ram = True
icbhi.ds_class = 'audio2img'


ptbxl = copy.deepcopy(mimic)
ptbxl.num_classes = 5
ptbxl.eval_classes = [0,1,2,3,4]
ptbxl.arch = 'ecg1d'
ptbxl.num_channels = 12
ptbxl.train_on = 'ptbxl_train'
ptbxl.valid_on = 'ptbxl_valid'
ptbxl.test_on = 'ludb_valid'
ptbxl.test_eval_classes = [0,1]
ptbxl.folds = [9,10]
ptbxl.preload_to_ram = True
ptbxl.ds_class = 'ecg'
ptbxl.num_epochs = 50
ptbxl.early_stop = 10
ptbxl.batch_size = 64
ptbxl.ngpu = 1


def get_args(which):
    expt_list = {}
    for name,x in globals().items():
        if isinstance(x,argparse.Namespace):
            x.name = name
            x.exp = name
            # add all defined items into expt_list
            expt_list[name] = x

            # create equivalent shuffled versions
            x_copy = copy.deepcopy(x)
            x_copy.name = '%s_shuffled'%(name)
            x_copy.exp = '%s_shuffled'%(name)
            x_copy.shuffle_pixels = True
            expt_list['%s_shuffled'%name] = x_copy

    return expt_list[which]

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_on', type=str, default=None)
    parser.add_argument('--valid_on', type=str, default=None)
    parser.add_argument('--test_on', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--disable_tqdm', type=str, default=None)
    parser.add_argument('--preload_to_ram', type=str, default=None)
    parser.add_argument('--ngpu', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--chkpt_every', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--shuffle_pixels', type=str, default=None)
    args = parser.parse_args()
    return args

def get_combined_args(args=None):
    cli_args  = vars(get_cli_args())
    if not args:
        args = vars(get_args(cli_args['exp']))
        
    for key in cli_args:
        val = cli_args[key]
        if val is not None:
            if isinstance(val,str):
                if val in ('True','TRUE','true'):
                    val = True
                elif val in ('False','FALSE','false'):
                    val = False
            args[key] = val
    args = argparse.Namespace(**args)
    return args

if __name__=='__main__':
    args = get_combined_args()    
    print(args)