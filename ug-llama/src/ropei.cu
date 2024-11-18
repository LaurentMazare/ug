using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
__device__ void ropei(const T * src, const T * cos, const T * sin, const int32_t * pos, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bth = idx / (d / 2);
    uint32_t i_d = idx - (d / 2) * i_bth;
    uint32_t i_t = (i_bth / h) % t;
    uint32_t i1 = i_bth * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    // TODO: This only works for consecutive timesteps?
    i_cs += pos[0] * d / 2;
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

extern "C" __global__ void ropei_f32(
    const float * src,
    const float * cos,
    const float * sin,
    const int * pos,
    float * dst,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t d
) {
  rope<float>(src, cos, sin, pos, dst, bh, td, d);
}

