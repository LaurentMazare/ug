use crate::runtime::{Slice, WithErr};
use cudarc::cublas::{GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DevicePtr};
use half::{bf16, f16};
use ug::{Layout, Result, Slice as _};

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;

    let lhs_stride = lhs_l.strides();
    let rhs_stride = rhs_l.strides();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        ug::bail!("non contiguous matmul, lhs: {lhs_l:?}, rhs: {rhs_l:?}")
    };
    // The b tensor has dims batching, m, k (lhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        ug::bail!("non contiguous matmul, lhs: {lhs_l:?}, rhs: {rhs_l:?}")
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => ug::bail!("non contiguous matmul, lhs: {lhs_l:?}, rhs: {rhs_l:?}"),
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => ug::bail!("non contiguous matmul, lhs: {lhs_l:?}, rhs: {rhs_l:?}"),
    };

    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

// Default for the reduced precision setting is false, similar to pytorch.
// https://github.com/pytorch/pytorch/issues/123157
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

unsafe fn gemm_strided_batched_f32(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f32>,
    a: &cudarc::driver::CudaView<f32>,
    b: &cudarc::driver::CudaView<f32>,
    c: &mut CudaSlice<f32>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let compute_type = if gemm_reduced_precision_f32() {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    };
    let alpha = &cfg.gemm.alpha as *const f32 as *const _;
    let beta = &cfg.gemm.beta as *const f32 as *const _;

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.lda,
        cfg.stride_a,
        *b.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        *c.device_ptr_mut() as *mut _,
        sys::cudaDataType_t::CUDA_R_32F,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
}

unsafe fn gemm_strided_batched_f16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<f16>,
    a: &cudarc::driver::CudaView<f16>,
    b: &cudarc::driver::CudaView<f16>,
    c: &mut CudaSlice<f16>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha = cfg.gemm.alpha;
    let beta = cfg.gemm.beta;
    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    let (compute_type, alpha, beta) = if gemm_reduced_precision_f16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            (&alpha) as *const f16 as *const _,
            (&beta) as *const f16 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.lda,
        cfg.stride_a,
        *b.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        *c.device_ptr_mut() as *mut _,
        sys::cudaDataType_t::CUDA_R_16F,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
}

unsafe fn gemm_strided_batched_bf16(
    cublas: &cudarc::cublas::CudaBlas,
    cfg: StridedBatchedConfig<bf16>,
    a: &cudarc::driver::CudaView<bf16>,
    b: &cudarc::driver::CudaView<bf16>,
    c: &mut CudaSlice<bf16>,
) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
    use cudarc::cublas::sys;
    use cudarc::driver::DevicePtrMut;

    let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
    let beta_f32: f32 = cfg.gemm.beta.to_f32();
    // The type for alpha and beta depends on the computeType.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
    let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    } else {
        (
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            (&alpha_f32) as *const f32 as *const _,
            (&beta_f32) as *const f32 as *const _,
        )
    };

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.lda,
        cfg.stride_a,
        *b.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.ldb,
        cfg.stride_b,
        beta,
        *c.device_ptr_mut() as *mut _,
        sys::cudaDataType_t::CUDA_R_16BF,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
        compute_type,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
}

pub(crate) fn matmul(
    blas: &cudarc::cublas::CudaBlas,
    dst: &mut Slice,
    lhs: &Slice,
    rhs: &Slice,
    bmnk: (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<()> {
    use crate::runtime::SliceInner::{BF16, F16, F32};
    let (dst_dt, lhs_dt, rhs_dt) = (dst.dtype(), lhs.dtype(), rhs.dtype());
    match (&mut dst.inner, &lhs.inner, &rhs.inner) {
        (BF16(dst), BF16(lhs), BF16(rhs)) => {
            let lhs = &lhs.slice(lhs_l.offset()..);
            let rhs = &rhs.slice(rhs_l.offset()..);
            let cfg = gemm_config(bf16::ONE, bf16::ZERO, bmnk, lhs_l, rhs_l)?;
            unsafe { gemm_strided_batched_bf16(blas, cfg, rhs, lhs, dst) }.w()?;
        }
        (F16(dst), F16(lhs), F16(rhs)) => {
            let lhs = &lhs.slice(lhs_l.offset()..);
            let rhs = &rhs.slice(rhs_l.offset()..);
            let cfg = gemm_config(f16::ONE, f16::ZERO, bmnk, lhs_l, rhs_l)?;
            unsafe { gemm_strided_batched_f16(blas, cfg, rhs, lhs, dst) }.w()?;
        }
        (F32(dst), F32(lhs), F32(rhs)) => {
            let lhs = &lhs.slice(lhs_l.offset()..);
            let rhs = &rhs.slice(rhs_l.offset()..);
            let cfg = gemm_config(1., 0., bmnk, lhs_l, rhs_l)?;
            unsafe { gemm_strided_batched_f32(blas, cfg, rhs, lhs, dst) }.w()?;
        }
        _ => {
            ug::bail!(
                "incorrect dtypes for matmul, dst: {dst_dt:?}, lhs: {lhs_dt:?}, rhs: {rhs_dt:?}"
            )
        }
    }
    Ok(())
}
