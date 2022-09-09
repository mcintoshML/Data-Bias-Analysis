from data_prep import prep_nih_data
from data_prep import prep_chexpert_data
from data_prep import prep_mimic_data
from data_prep import prep_audio_data
from data_prep import prep_covid_internal_data
from data_prep import prep_covid_kaggle_data
from data_prep import prep_ild_data
from data_prep import prep_ecg_data
from networks import arch
from dataloaders import radiology_2d
from dataloaders import ild_3d
from dataloaders import audio2img
from dataloaders import ecg_1d
from analyze_results import plot_auc_graphs


if __name__=='__main__':

    # -- Preprocess and save for faster disc access --
    # prep_nih_data.main()
    # prep_chexpert_data.main()
    # prep_mimic_data.main()
    # prep_covid_internal_data.main()
    # prep_covid_kaggle_data.main()
    # prep_ild_data.main()
    # prep_audio_data.main()
    # prep_ecg_data.main()

    # -- Create model architectures for use --
    # arch.main()

    # -- Dataloaders -- 
    # -- Normalization and Randomization done within dataset classes --
    # radiology_2d.main()
    # ild_3d.main()
    # audio2img.main()
    # ecg_1d.main()

    # -- Train and Save outputs to file -- 
    # See run_expts.sh for examples
    
    # -- data acquisition bias calculations -- 
    # -- calibrate and project --
    # plot_auc_graphs.main()

    pass

