#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void chroma_custom_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_frames,
    int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_frames * num_bins;

    if (idx < total_elements) {
        output[idx] = input[idx] * 1.0f;
    }
}

torch::Tensor launch_chroma_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor mora biti na GPU-u!");
    TORCH_CHECK(input.is_contiguous(), "Input tensor mora biti kontiguan!");

    int num_frames = input.size(0);
    int num_bins = input.size(1);

    auto output = torch::empty_like(input);

    int total_elements = num_frames * num_bins;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    chroma_custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_frames,
        num_bins
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_chroma_cuda, "Chroma CUDA kernel forward");
}