## --------------- Individual X-Rays Expts ---------------
# exp_set="chexpert_80_20 chexpert_80_20_shuffled mimic mimic_shuffled"
# test_datasets="chexpert_80_20 mimic nih"

## --------------- Combined X-Ray Expts ---------------
# exp_set="combined combined_nihmimic combined_shuffled combined_nihmimic_shuffled"
# test_datasets="combined combined_nihmimic nih chexpert_80_20"

## --------------- Covid X-Ray Expts ---------------
# exp_set="covid_kaggle covid_kaggle_shuffled"
# test_datasets="covid_kaggle covid_internal"

## --------------- ILD 3D Expts ---------------
# exp_set="ild3d ild3d_shuffled"
# test_datasets="ild_TR ild_plan"

## --------------- Audio 1D as Mel-SPectrogram Expts ---------------
exp_set="icbhi icbhi_shuffled"
test_datasets="icbhi just"

## --------------- ECG 1D Expts ---------------
# exp_set="ptbxl ptbxl_shuffled"
# test_datasets="ptbxl ludb"

for exp in $exp_set;do
    CUDA_VISIBLE_DEVICES=1 python experiments.py --exp "${exp}" --task train
    for test_on in $test_datasets;do
        CUDA_VISIBLE_DEVICES=1 python experiments.py --exp "${exp}" --task eval --test_on "${test_on}_valid"
    done
done