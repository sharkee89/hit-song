import os
from torch.utils.cpp_extension import load

# Putanja do foldera
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_file = os.path.join(current_dir, "chroma_kernel.cu")

# JIT (Just-In-Time) kompajliranje CUDA ekstenzije pri prvom pokretanju
chroma_cuda_ext = load(
    name="chroma_cuda_ext",
    sources=[cuda_file],
    verbose=True
)

def run_custom_cuda_chroma(tensor_input):
    return chroma_cuda_ext.forward(tensor_input)