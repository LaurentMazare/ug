using uint32_t = unsigned int;
using int32_t = int;

template<typename T>
METAL_FUNC void softmax(
    constant uint32_t & src_numel,
    constant uint32_t & el_to_sum_per_block,
    device const T * src,
    device T * dst,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    uint32_t start_idx = dst_id * el_to_sum_per_block;
    uint32_t stop_idx = metal::min(start_idx + el_to_sum_per_block, src_numel);
    uint32_t idx = start_idx + tid;

    float tmp = -INFINITY;
    while (idx < stop_idx) {
        tmp = metal::max(tmp, float(src[idx]));
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = metal::max(shared_memory[tid], shared_memory[tid + s]);\
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float _max = shared_memory[0];

    /* prevent tid=0 from overwriting _max before other threads have written it */
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    shared_memory[tid] = 0;

    idx = start_idx + tid;
    while (idx < stop_idx) {
        const float val = metal::exp(float(src[idx]) - _max);
        dst[idx] = T(val);
        shared_memory[tid] += val;
        idx += block_dim;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    const T inv_acc = T(1.0 / shared_memory[0]);
    idx = start_idx + tid;
    while (idx < stop_idx) {
        dst[idx] *= inv_acc;
        idx += block_dim;
    }
}

kernel void softmax_f32(
    constant uint32_t &src_numel,
    constant uint32_t &el_to_sum_per_block,
    device const float *src,
    device float *dst,
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    threadgroup float shared_memory[2048];
    shared_memory[tid] = -INFINITY;
    softmax<float>(src_numel, el_to_sum_per_block, src, dst, id, tid, dst_id, block_dim, shared_memory);
}
