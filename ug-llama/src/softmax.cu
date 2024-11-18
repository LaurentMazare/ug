using uint32_t = unsigned int;
using int32_t = int;

// TODO: the code below uses max and exp so cannot be generalized to f16/bf16 out of the box.
// Softmax implementation adapted from ggml.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L4159
template <typename T, typename ACC>
__device__ void softmax(const T * x, T * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    T max_val = -1.0 / 0.0;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = max(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }

    ACC tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const T val = exp(x[i] - max_val);
        tmp += static_cast<ACC>(val);
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const ACC inv_tmp = 1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }
}

extern "C" __global__ void softmax_f32(
    const float *src,
    float *dst,
    const int32_t ncols) {
  softmax<float, float>(src, dst, ncols);
}
