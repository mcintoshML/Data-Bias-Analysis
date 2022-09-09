* Typical Install Time : 15 minutes
* Expected run time 
    * Experiments run in under 24 hours with batch size = 16 for larger chest X-Ray datasets
    * Smaller dataset experiments complete in under 3 hours
    * Estimates are using single NVIDIA A100 Tensor Core GPU
* Expected output
    * Logs are print out during the training process 
    * Completion of experiments running `run_expts.sh` and `analyze_results/plot_auc_graphs.py` result in AUC projection graphs genereated into `analyze_results/auc_graphs/`
* Full example which runs in under 1 hour provided in `example_code/`. Use `run_example.py` to run the example setup.