use crate::runtime::WithErr;
use cudarc::cublas::{GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr};
use half::{bf16, f16};
use ug::Result;

pub trait CudaType: ug::WithDType + DeviceRepr {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError>;
}

#[allow(clippy::too_many_arguments)]
pub fn gemm<T: CudaType>(
    cublas: &cudarc::cublas::CudaBlas,
    dst: &mut cudarc::driver::CudaSlice<T>,
    lhs: (&cudarc::driver::CudaSlice<T>, usize),
    rhs: (&cudarc::driver::CudaSlice<T>, usize),
    m: usize,
    n: usize,
    k: usize,
    lhs_b: usize,
    b_stride: usize,
    (_dst_cs, dst_rs): (usize, usize),
    (lhs_m1, lhs_m2): (usize, usize),
    (rhs_m1, rhs_m2): (usize, usize),
) -> Result<()> {
    // TODO: Check that the parameters are valid, e.g. dst_cs == 1 etc.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        ug::bail!("non-contiguous matmul rhs m:{m} n:{n} k:{k} {rhs_m2} {rhs_m1}")
    };
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        ug::bail!("non-contiguous matmul lhs m:{m} n:{n} k:{k} {lhs_m2} {lhs_m1}")
    };

    let gemm = GemmConfig {
        alpha: T::one(),
        beta: T::zero(),
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: dst_rs as i32,
        transa,
        transb,
    };
    let cfg = StridedBatchedConfig {
        batch_size: lhs_b as i32,
        gemm,
        stride_a: b_stride as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };
    let lhs = &lhs.0.slice(lhs.1..);
    let rhs = &rhs.0.slice(rhs.1..);
    unsafe { T::gemm(cublas, cfg, rhs, lhs, dst).w()? };
    Ok(())
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

impl CudaType for f32 {
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
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
}

impl CudaType for f16 {
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
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
}

impl CudaType for bf16 {
    unsafe fn gemm(
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
        let alpha = f16::from_f32(alpha_f32);
        let beta = f16::from_f32(beta_f32);
        // The type for alpha and beta depends on the computeType.
        // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
        let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
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
}
