import torch
from torch import cuda, device

# CUDA for PyTorch
use_cuda = cuda.is_available()
if use_cuda:
    device2 = device("cuda")

else:
    device2 = device("cpu")

torch.set_default_tensor_type('torch.FloatTensor')

