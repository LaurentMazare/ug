using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
__device__ void cat(
    const T * lhs,
    const T * rhs,
    T * dst,
    const uint32_t d1,
    const uint32_t d2_l,
    const uint32_t d2_r,
    const uint32_t d2_lr
) {
  const uint32_t i1 = blockIdx.x;
  const uint32_t i2 = threadIdx.x;
  const uint32_t block_size = blockDim.x;

  for (uint32_t v = i2; v < d2_l; v += block_size) {
    dst[i1 * d2_lr + v] = lhs[i1 * d2_l + v];
  }
  for (uint32_t v = i2; v < d2_r; v += block_size) {
    dst[i1 * d2_lr + d2_l + v] = rhs[i1 * d2_r + v];
  }
}

extern "C" __global__ void cat_f32(
    const float * lhs,
    const float * rhs,
    float * dst,
    const uint32_t d1,
    const uint32_t d2_l,
    const uint32_t d2_r,
    const uint32_t d2_lr
) {
  cat<float>(lhs, rhs, dst, d1, d2_l, d2_r, d2_lr);
}
