use cudarc::driver::DeviceRepr;
pub use cudarc::driver::{CudaFunction, DeviceSlice, LaunchConfig};
use std::sync::Arc;
use ug::{Device as D, Error, Result, Slice as S, WithDType};

pub trait WithErr {
    type T;
    fn w(self) -> Result<Self::T>;
}

impl<T> WithErr for std::result::Result<T, cudarc::cublas::result::CublasError> {
    type T = T;
    fn w(self) -> Result<Self::T> {
        self.map_err(|v| Error::wrap(v).bt())
    }
}

impl<T> WithErr for std::result::Result<T, cudarc::driver::DriverError> {
    type T = T;
    fn w(self) -> Result<Self::T> {
        self.map_err(|v| Error::wrap(v).bt())
    }
}

impl<T> WithErr for std::result::Result<T, cudarc::nvrtc::CompileError> {
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

#[derive(Clone)]
pub struct Func {
    func: CudaFunction,
    cfg: LaunchConfig,
}

impl Func {
    pub fn new(func: CudaFunction, cfg: LaunchConfig) -> Self {
        Self { func, cfg }
    }
}

macro_rules! impl_launch { ($name:ident, [$($Vars:tt),*]) => {
    pub unsafe fn $name<$($Vars: DeviceRepr),*>(
        &self,
        args: ($($Vars, )*)
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;
        let func = self.func.clone();
        unsafe { func.launch(self.cfg, args).w()? };
        Ok(())
    }
}
}

#[allow(clippy::missing_safety_doc)]
impl Func {
    impl_launch!(launch1, [P1]);
    impl_launch!(launch2, [P1, P2]);
    impl_launch!(launch3, [P1, P2, P3]);
    impl_launch!(launch4, [P1, P2, P3, P4]);
    impl_launch!(launch5, [P1, P2, P3, P4, P5]);
    impl_launch!(launch6, [P1, P2, P3, P4, P5, P6]);
    impl_launch!(launch7, [P1, P2, P3, P4, P5, P6, P7]);
    impl_launch!(launch8, [P1, P2, P3, P4, P5, P6, P7, P8]);
}

#[derive(Debug, Clone)]
pub struct Device {
    device: Arc<cudarc::driver::CudaDevice>,
    blas: Arc<cudarc::cublas::CudaBlas>,
}

// A GADT based solution would seem better than this variant but not sure how to do this in rust.
#[derive(Debug, Clone)]
pub enum SliceInner {
    F32(cudarc::driver::CudaSlice<f32>),
    F16(cudarc::driver::CudaSlice<half::f16>),
    BF16(cudarc::driver::CudaSlice<half::bf16>),
    I32(cudarc::driver::CudaSlice<i32>),
    I64(cudarc::driver::CudaSlice<i64>),
}

#[derive(Debug, Clone)]
pub struct Slice {
    pub(crate) inner: SliceInner,
    device: Device,
}

pub trait ToSlice: Sized {
    fn slice(s: &Slice) -> Result<&cudarc::driver::CudaSlice<Self>>;
}

macro_rules! to_slice {
    ($ty:ty, $dtype:ident) => {
        impl ToSlice for $ty {
            fn slice(s: &Slice) -> Result<&cudarc::driver::CudaSlice<Self>> {
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

impl Slice {
    pub fn slice<D: ToSlice>(&self) -> Result<&cudarc::driver::CudaSlice<D>> {
        ToSlice::slice(self)
    }
}

unsafe impl cudarc::driver::safe::DeviceRepr for &Slice {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        match &self.inner {
            SliceInner::BF16(v) => v.as_kernel_param(),
            SliceInner::F16(v) => v.as_kernel_param(),
            SliceInner::F32(v) => v.as_kernel_param(),
            SliceInner::I32(v) => v.as_kernel_param(),
            SliceInner::I64(v) => v.as_kernel_param(),
        }
    }
}

unsafe impl cudarc::driver::safe::DeviceRepr for &mut Slice {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        match &self.inner {
            SliceInner::BF16(v) => v.as_kernel_param(),
            SliceInner::F16(v) => v.as_kernel_param(),
            SliceInner::F32(v) => v.as_kernel_param(),
            SliceInner::I32(v) => v.as_kernel_param(),
            SliceInner::I64(v) => v.as_kernel_param(),
        }
    }
}

impl Device {
    pub fn new(device_index: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(device_index).w()?;
        let blas = cudarc::cublas::CudaBlas::new(device.clone()).w()?;
        Ok(Self { device, blas: Arc::new(blas) })
    }

    pub fn cudarc_device(&self) -> &cudarc::driver::CudaDevice {
        self.device.as_ref()
    }

    pub fn blas(&self) -> &cudarc::cublas::CudaBlas {
        self.blas.as_ref()
    }

    pub fn compile_cu(
        &self,
        cu_code: &str,
        module_name: &str,
        func_name: &'static str,
    ) -> Result<CudaFunction> {
        let opts =
            cudarc::nvrtc::CompileOptions { use_fast_math: Some(true), ..Default::default() };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cu_code, opts).w()?;
        self.device.load_ptx(ptx, module_name, &[func_name]).w()?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => ug::bail!("unknown function {module_name}::{func_name}"),
        };
        Ok(func)
    }

    pub fn compile_ptx(
        &self,
        ptx_code: &str,
        module_name: &str,
        func_name: &'static str,
    ) -> Result<CudaFunction> {
        let ptx = cudarc::nvrtc::safe::Ptx::from_src(ptx_code);
        self.device.load_ptx(ptx, module_name, &[func_name]).w()?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => ug::bail!("unknown function {module_name}::{func_name}"),
        };
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
        self.device.synchronize().w()?;
        Ok(())
    }
}

impl ug::Device for Device {
    type Slice = Slice;
    type Func = Func;

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

    fn compile(&self, kernel: &ug::lang::ssa::Kernel, name: Option<&str>) -> Result<Self::Func> {
        let mut cu_code = Vec::with_capacity(8192);
        let kernel_id = KernelId::new().as_usize();
        let func_name = match name {
            Some(name) => &format!("ug_{name}_{kernel_id}"),
            None => &format!("ug_{kernel_id}"),
        };
        crate::code_gen::gen(&mut cu_code, func_name, kernel)?;
        let func_name_s = Box::leak(Box::new(func_name.to_string()));
        let cu_code = String::from_utf8(cu_code)?;
        let cfg = kernel.launch_config();
        let cfg = LaunchConfig {
            grid_dim: (cfg.grid_dim, 1, 1),
            block_dim: (cfg.block_dim, 1, 1),
            shared_mem_bytes: cfg.shared_mem,
        };
        let func = self.compile_cu(&cu_code, func_name, func_name_s)?;
        Ok(Func { func, cfg })
    }

    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()> {
        use cudarc::driver::LaunchAsync;
        let func = f.func.clone();
        // TODO: Avoid these.
        if f.cfg.block_dim.0 == 0
            || f.cfg.block_dim.1 == 0
            || f.cfg.block_dim.2 == 0
            || f.cfg.grid_dim.0 == 0
            || f.cfg.grid_dim.1 == 0
            || f.cfg.grid_dim.2 == 0
        {
            return Ok(());
        }
        match args {
            [a1] => {
                unsafe { func.launch(f.cfg, (&**a1,)).w()? };
            }
            [a1, a2] => {
                unsafe { func.launch(f.cfg, (&**a1, &**a2)).w()? };
            }
            [a1, a2, a3] => {
                unsafe { func.launch(f.cfg, (&**a1, &**a2, &**a3)).w()? };
            }
            [a1, a2, a3, a4] => {
                unsafe { func.launch(f.cfg, (&**a1, &**a2, &**a3, &**a4)).w()? };
            }
            [a1, a2, a3, a4, a5] => {
                unsafe { func.launch(f.cfg, (&**a1, &**a2, &**a3, &**a4, &**a5)).w()? };
            }
            _ => ug::bail!("unsupported number of args for kernel {}", args.len()),
        }
        Ok(())
    }

    fn matmul(
        &self,
        dst: &mut Self::Slice,
        lhs: &Self::Slice,
        rhs: &Self::Slice,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &ug::Layout,
        rhs_l: &ug::Layout,
    ) -> Result<()> {
        crate::gemm::matmul(&self.blas, dst, lhs, rhs, bmnk, lhs_l, rhs_l)
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

    fn copy_host_to_device<DT: WithDType>(&mut self, src: &[DT]) -> Result<()> {
        use ug::CpuStorageRef as C;
        use SliceInner as S;
        match (&mut self.inner, DT::to_cpu_storage(src)) {
            (S::BF16(dst), C::BF16(src)) => self.device.device.htod_sync_copy_into(src, dst).w()?,
            (S::F16(dst), C::F16(src)) => self.device.device.htod_sync_copy_into(src, dst).w()?,
            (S::F32(dst), C::F32(src)) => self.device.device.htod_sync_copy_into(src, dst).w()?,
            (S::I32(dst), C::I32(src)) => self.device.device.htod_sync_copy_into(src, dst).w()?,
            (S::I64(dst), C::I64(src)) => self.device.device.htod_sync_copy_into(src, dst).w()?,
            (_, _) => ug::bail!("htod dtype mismatch, dst {:?}, src {:?}", self.dtype(), DT::DTYPE),
        }
        Ok(())
    }

    fn copy_device_to_host<DT: WithDType>(&self, dst: &mut [DT]) -> Result<()> {
        use ug::CpuStorageRefMut as C;
        use SliceInner as S;
        match (&self.inner, DT::to_cpu_storage_mut(dst)) {
            (S::BF16(src), C::BF16(dst)) => self.device.device.dtoh_sync_copy_into(src, dst).w()?,
            (S::F16(src), C::F16(dst)) => self.device.device.dtoh_sync_copy_into(src, dst).w()?,
            (S::F32(src), C::F32(dst)) => self.device.device.dtoh_sync_copy_into(src, dst).w()?,
            (S::I32(src), C::I32(dst)) => self.device.device.dtoh_sync_copy_into(src, dst).w()?,
            (S::I64(src), C::I64(dst)) => self.device.device.dtoh_sync_copy_into(src, dst).w()?,
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
