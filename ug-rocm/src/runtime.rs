// TEAM-510: Updated to use rocm-rs instead of cudarc
pub use rocm_rs::hip::utils::Dim3;
pub use rocm_rs::hip::{DeviceMemory, Function as HipFunction, Stream as HipStream};
use std::sync::Arc;
use ug::{Device as D, Error, Result, Slice as S, WithDType};

pub trait WithErr {
    type T;
    fn w(self) -> Result<Self::T>;
}

// TEAM-510: Updated error handling for ROCm
impl<T> WithErr for std::result::Result<T, rocm_rs::hip::error::Error> {
    type T = T;
    fn w(self) -> Result<Self::T> {
        self.map_err(|v| Error::wrap(v).bt())
    }
}

impl<T> WithErr for std::result::Result<T, rocm_rs::rocblas::error::Error> {
    type T = T;
    fn w(self) -> Result<Self::T> {
        self.map_err(|v| Error::wrap(v).bt())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(usize);

impl KernelId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

// TEAM-510: HIP function wrapper (similar to CUDA's Func)
#[derive(Clone)]
pub struct Func {
    func: HipFunction,
    stream: Arc<HipStream>,
    grid_dim: Dim3,
    block_dim: Dim3,
}

// TEAM-510: HIP function implementation
impl Func {
    pub fn new(device: &crate::Device, func: HipFunction, grid_dim: Dim3, block_dim: Dim3) -> Self {
        Self { func, stream: device.hip_stream().clone(), grid_dim, block_dim }
    }

    pub fn launch(&self, kernel_params: &mut [*mut std::ffi::c_void]) -> Result<()> {
        self.func
            .launch(
                self.grid_dim,
                self.block_dim,
                0, // shared_mem_bytes
                Some(&self.stream),
                kernel_params,
            )
            .w()
    }

    pub fn grid_dim(&self) -> Dim3 {
        self.grid_dim
    }

    pub fn block_dim(&self) -> Dim3 {
        self.block_dim
    }
}

#[macro_export]
macro_rules! bargs {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

// TEAM-510: ROCm device wrapper (similar to CUDA's Device)
#[derive(Debug, Clone)]
pub struct Device {
    device: Arc<rocm_rs::hip::Device>,
    stream: Arc<rocm_rs::hip::Stream>,
    blas: Arc<rocm_rs::rocblas::RocBlas>,
}

// TEAM-510: ROCm device memory slices (similar to CUDA's SliceInner)
// A GADT based solution would seem better than this variant but not sure how to do this in rust.
#[derive(Debug, Clone)]
pub enum SliceInner {
    F32(rocm_rs::hip::DeviceMemory<f32>),
    F16(rocm_rs::hip::DeviceMemory<half::f16>),
    BF16(rocm_rs::hip::DeviceMemory<half::bf16>),
    I32(rocm_rs::hip::DeviceMemory<i32>),
    I64(rocm_rs::hip::DeviceMemory<i64>),
}

#[derive(Debug, Clone)]
pub struct Slice {
    pub(crate) inner: SliceInner,
    device: Device,
}

// TEAM-510: ROCm ToSlice trait (similar to CUDA but uses DeviceMemory)
pub trait ToSlice: Sized {
    fn slice(s: &Slice) -> Result<&rocm_rs::hip::DeviceMemory<Self>>;
}

macro_rules! to_slice {
    ($ty:ty, $dtype:ident) => {
        impl ToSlice for $ty {
            fn slice(s: &Slice) -> Result<&rocm_rs::hip::DeviceMemory<Self>> {
                match &s.inner {
                    SliceInner::$dtype(s) => Ok(s),
                    _ => ug::bail!(
                        "dtype mismatch, expected {:?}, got {:?}",
                        ug::DType::$dtype,
                        s.dtype()
                    ),
                }
            }
        }
    };
}
to_slice!(f32, F32);
to_slice!(half::f16, F16);
to_slice!(half::bf16, BF16);
to_slice!(i32, I32);
to_slice!(i64, I64);

// TEAM-510: Slice helper method
impl Slice {
    pub fn slice<D: ToSlice>(&self) -> Result<&rocm_rs::hip::DeviceMemory<D>> {
        ToSlice::slice(self)
    }
}

// TEAM-510: Device pointer access for ROCm
impl Slice {
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        match &self.inner {
            SliceInner::BF16(v) => v.as_ptr() as *const std::ffi::c_void,
            SliceInner::F16(v) => v.as_ptr() as *const std::ffi::c_void,
            SliceInner::F32(v) => v.as_ptr() as *const std::ffi::c_void,
            SliceInner::I32(v) => v.as_ptr() as *const std::ffi::c_void,
            SliceInner::I64(v) => v.as_ptr() as *const std::ffi::c_void,
        }
    }

    pub fn device_ptr_mut(&mut self) -> *mut std::ffi::c_void {
        match &mut self.inner {
            SliceInner::BF16(v) => v.as_mut_ptr() as *mut std::ffi::c_void,
            SliceInner::F16(v) => v.as_mut_ptr() as *mut std::ffi::c_void,
            SliceInner::F32(v) => v.as_mut_ptr() as *mut std::ffi::c_void,
            SliceInner::I32(v) => v.as_mut_ptr() as *mut std::ffi::c_void,
            SliceInner::I64(v) => v.as_mut_ptr() as *mut std::ffi::c_void,
        }
    }
}

// TEAM-510: Device implementation for ROCm
impl Device {
    pub fn new(device_index: usize) -> Result<Self> {
        let device = Arc::new(rocm_rs::hip::Device::new(device_index as i32).w()?);
        let stream = Arc::new(rocm_rs::hip::Stream::new().w()?);
        let blas = Arc::new(rocm_rs::rocblas::RocBlas::new().w()?);
        Ok(Self { device, stream, blas })
    }

    pub fn hip_stream(&self) -> &Arc<rocm_rs::hip::Stream> {
        &self.stream
    }

    pub fn blas(&self) -> &rocm_rs::rocblas::RocBlas {
        self.blas.as_ref()
    }

    pub fn compile_hip(&self, hip_code: &str, func_name: &str) -> Result<HipFunction> {
        // Compile HIP code using rocm-rs
        let module = rocm_rs::hip::module::compile_and_load(hip_code, &[]).w()?;
        let func = module.get_function(func_name).w()?;
        Ok(func)
    }

    pub fn compile_hsaco(&self, hsaco_code: &[u8], func_name: &'static str) -> Result<HipFunction> {
        // Load pre-compiled HSACO binary
        let module = self.device.load_module(hsaco_code).w()?;
        let func = module.get_function(func_name).w()?;
        Ok(func)
    }

    pub fn zeros(&self, len: usize) -> Result<Slice> {
        let slice = self.device.alloc_zeros::<f32>(len).w()?;
        Ok(Slice { inner: SliceInner::F32(slice), device: self.clone() })
    }

    pub fn slice_from_values<D: WithDType>(&self, vs: &[D]) -> Result<Slice> {
        let mut slice = unsafe { self.allocate_uninit(D::DTYPE, vs.len())? };
        slice.copy_host_to_device(vs)?;
        Ok(slice)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().w()?;
        Ok(())
    }
}

impl ug::Device for Device {
    type Slice = Slice;
    type Func = Func;

    // TEAM-510: Allocate device memory for ROCm
    #[allow(clippy::missing_transmute_annotations)]
    unsafe fn allocate_uninit(&self, dtype: ug::DType, len: usize) -> Result<Self::Slice> {
        let inner = match dtype {
            ug::DType::F32 => {
                let slice = self.device.alloc::<f32>(len).w()?;
                SliceInner::F32(slice)
            }
            ug::DType::F16 => {
                let slice = self.device.alloc::<half::f16>(len).w()?;
                SliceInner::F16(slice)
            }
            ug::DType::BF16 => {
                let slice = self.device.alloc::<half::bf16>(len).w()?;
                SliceInner::BF16(slice)
            }
            ug::DType::I32 => {
                let slice = self.device.alloc::<i32>(len).w()?;
                SliceInner::I32(slice)
            }
            ug::DType::I64 => {
                let slice = self.device.alloc::<i64>(len).w()?;
                SliceInner::I64(slice)
            }
        };
        Ok(Slice { inner, device: self.clone() })
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }

    // TEAM-510: Compile ug kernel to HIP code
    fn compile(&self, kernel: &ug::lang::ssa::Kernel, name: Option<&str>) -> Result<Self::Func> {
        let mut hip_code = Vec::with_capacity(8192);
        let kernel_id = KernelId::new().as_usize();
        let func_name = match name {
            Some(name) => format!("ug_{name}_{kernel_id}"),
            None => format!("ug_{kernel_id}"),
        };
        crate::code_gen::gen(&mut hip_code, &func_name, kernel)?;
        let hip_code = String::from_utf8(hip_code)?;
        let cfg = kernel.launch_config();
        let grid_dim = Dim3 { x: cfg.grid_dim as u32, y: 1, z: 1 };
        let block_dim = Dim3 { x: cfg.block_dim as u32, y: 1, z: 1 };
        let func = self.compile_hip(&hip_code, &func_name)?;
        Ok(Func::new(self, func, grid_dim, block_dim))
    }

    // TEAM-510: Run HIP kernel with arguments
    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()> {
        // Check for zero dimensions
        if f.block_dim().x == 0
            || f.block_dim().y == 0
            || f.block_dim().z == 0
            || f.grid_dim().x == 0
            || f.grid_dim().y == 0
            || f.grid_dim().z == 0
        {
            return Ok(());
        }

        // Build kernel parameters
        let mut kernel_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(args.len());
        for arg in args.iter_mut() {
            kernel_params.push(arg.device_ptr_mut());
        }

        // Launch kernel
        f.launch(&mut kernel_params)?;
        Ok(())
    }

    // TEAM-510: Matrix multiplication using rocBLAS
    fn matmul(
        &self,
        _dst: &mut Self::Slice,
        _lhs: &Self::Slice,
        _rhs: &Self::Slice,
        _bmnk: (usize, usize, usize, usize),
        _lhs_l: &ug::Layout,
        _rhs_l: &ug::Layout,
    ) -> Result<()> {
        // TODO: Implement rocBLAS matrix multiplication
        ug::bail!("matmul not yet implemented for ROCm")
    }

    fn use_grid() -> bool {
        true
    }
}

impl ug::Slice for Slice {
    type Device = Device;
    fn dtype(&self) -> ug::DType {
        match &self.inner {
            SliceInner::F32(_) => ug::DType::F32,
            SliceInner::F16(_) => ug::DType::F16,
            SliceInner::BF16(_) => ug::DType::BF16,
            SliceInner::I32(_) => ug::DType::I32,
            SliceInner::I64(_) => ug::DType::I64,
        }
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    // TEAM-510: Copy from host to device memory
    fn copy_host_to_device<DT: WithDType>(&mut self, src: &[DT]) -> Result<()> {
        use ug::CpuStorageRef as C;
        use SliceInner as S;
        match (&mut self.inner, DT::to_cpu_storage(src)) {
            (S::BF16(dst), C::BF16(src)) => dst.copy_from_host(src).w()?,
            (S::F16(dst), C::F16(src)) => dst.copy_from_host(src).w()?,
            (S::F32(dst), C::F32(src)) => dst.copy_from_host(src).w()?,
            (S::I32(dst), C::I32(src)) => dst.copy_from_host(src).w()?,
            (S::I64(dst), C::I64(src)) => dst.copy_from_host(src).w()?,
            (_, _) => ug::bail!("htod dtype mismatch, dst {:?}, src {:?}", self.dtype(), DT::DTYPE),
        }
        Ok(())
    }

    // TEAM-510: Copy from device to host memory
    fn copy_device_to_host<DT: WithDType>(&self, dst: &mut [DT]) -> Result<()> {
        use ug::CpuStorageRefMut as C;
        use SliceInner as S;
        match (&self.inner, DT::to_cpu_storage_mut(dst)) {
            (S::BF16(src), C::BF16(dst)) => src.copy_to_host(dst).w()?,
            (S::F16(src), C::F16(dst)) => src.copy_to_host(dst).w()?,
            (S::F32(src), C::F32(dst)) => src.copy_to_host(dst).w()?,
            (S::I32(src), C::I32(dst)) => src.copy_to_host(dst).w()?,
            (S::I64(src), C::I64(dst)) => src.copy_to_host(dst).w()?,
            (_, _) => ug::bail!("dtoh dtype mismatch, dst {:?}, src {:?}", DT::DTYPE, self.dtype()),
        }
        Ok(())
    }

    fn len(&self) -> usize {
        match &self.inner {
            SliceInner::F32(slice) => slice.len(),
            SliceInner::F16(slice) => slice.len(),
            SliceInner::BF16(slice) => slice.len(),
            SliceInner::I32(slice) => slice.len(),
            SliceInner::I64(slice) => slice.len(),
        }
    }
}
