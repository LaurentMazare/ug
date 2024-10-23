template <typename T>
static __device__ __forceinline__ T warp_reduce_sum(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T warp_reduce_max(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = max(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T warp_reduce_min(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = min(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_sum(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_sum(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_sum(x);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_max(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_max(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_max(x);
    }
    return x;
}

template <typename T>
static __device__ __forceinline__ T block_reduce_min(T x) {
    const int block_size = blockDim.x;
    x = warp_reduce_min(x);
    if (block_size > 32) {
        __shared__ T smem[32];
        unsigned int warp_id = threadIdx.x / 32;
        unsigned int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            smem[warp_id] = x;
        }
        __syncthreads();
        x = smem[lane_id];
        x = warp_reduce_min(x);
    }
    return x;
}
