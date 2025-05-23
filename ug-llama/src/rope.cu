using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
__device__ void rope(const T * src, const T * cos, const T * sin, const int32_t * pos, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bh = idx / (td / 2);
    uint32_t i_td = idx - (td / 2) * i_bh;
    uint32_t i_t = i_td / (d / 2);
    uint32_t i_d = i_td - (d / 2) * i_t;
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    // TODO: This only works for consecutive timesteps?
    T c = cos[pos[0] * d / 2 + i_cs];
    T s = sin[pos[0] * d / 2 + i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

extern "C" __global__ void rope_f32(
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
