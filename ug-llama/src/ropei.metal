using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
METAL_KERNEL void ropei(
    device const T * src,
    device const T * cos,
    device const T * sin,
    device const int32_t * pos,
    device T * dst,
    constant uint32_t bh,
    constant uint32_t td,
    constant uint32_t d,
    uint32_t idx
) {
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

kernel void ropei_f32(
    device const float * src,
    device const float * cos,
    device const float * sin,
    device const int * pos,
    device float * dst,
    constant uint32_t &bh,
    constant uint32_t &td,
    constant uint32_t &d,
    uint tid [[ thread_position_in_grid ]]
) {
  ropei<float>(src, cos, sin, pos, dst, bh, td, d, tid);
}
