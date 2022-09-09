from engine import sup_classifier 
from engine.arg_handler import get_combined_args
from engine import utils
import torch

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


if __name__=='__main__':
    # Gets CLI args and overwrites preset params
    args = get_combined_args()

    # Context manager - sets with the seed within this block for numpy, torch, python
    with utils.set_seed(args.seed):
        exp = sup_classifier.ExperimentRunner(args)
        if args.task == 'train':
            exp.train(resume='new')
        elif args.task == 'resume':
            exp.train(resume='last')
        elif args.task == 'eval':
            exp.test(test_on=args.test_on, replace=True)
        elif args.task == 'check_and_eval':
            exp.test(test_on=args.test_on, replace=False)