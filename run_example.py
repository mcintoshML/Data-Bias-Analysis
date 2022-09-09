import torch
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

from example_code import example_train
from example_code import example_dataset
from example_code import example_plot


if __name__=='__main__':
   example_dataset.main()
   example_train.main()
   example_plot.main()
