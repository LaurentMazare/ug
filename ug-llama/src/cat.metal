using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
METAL_FUNC void cat(
    device const T * lhs,
    device const T * rhs,
    device T * dst,
    constant uint32_t &d1,
    constant uint32_t &d2_l,
    constant uint32_t &d2_r,
    constant uint32_t &d2_lr,
    uint32_t i1,
    uint32_t i2,
    uint32_t block_size
) {
  for (uint32_t v = i2; v < d2_l; v += block_size) {
    dst[i1 * d2_lr + v] = lhs[i1 * d2_l + v];
  }
  for (uint32_t v = i2; v < d2_r; v += block_size) {
    dst[i1 * d2_lr + d2_l + v] = rhs[i1 * d2_r + v];
  }
}

kernel void cat_f32(
    device const float * lhs,
    device const float * rhs,
    device float * dst,
    constant uint32_t &d1,
    constant uint32_t &d2_l,
    constant uint32_t &d2_r,
    constant uint32_t &d2_lr,
    uint i1 [[threadgroup_position_in_grid]],
    uint i2 [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
  cat<float>(lhs, rhs, dst, d1, d2_l, d2_r, d2_lr, i1, i2, block_size);
}
