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


