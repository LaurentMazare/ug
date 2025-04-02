pub use cudarc::driver::{
    CudaFunction, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, LaunchConfig,
};
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
    stream: Arc<CudaStream>,
    cfg: LaunchConfig,
}

impl Func {
    pub fn new(stream: Arc<CudaStream>, func: CudaFunction, cfg: LaunchConfig) -> Self {
        Self { func, stream, cfg }
    }

    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }

    pub fn launch_cfg(&self) -> &LaunchConfig {
        &self.cfg
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

#[derive(Debug, Clone)]
pub struct Device {
    context: Arc<cudarc::driver::CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
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

impl Slice {
    pub fn device_ptr<'a>(
        &'a self,
        stream: &'a Arc<CudaStream>,
    ) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
        match &self.inner {
            SliceInner::BF16(v) => v.device_ptr(stream),
            SliceInner::F16(v) => v.device_ptr(stream),
            SliceInner::F32(v) => v.device_ptr(stream),
            SliceInner::I32(v) => v.device_ptr(stream),
            SliceInner::I64(v) => v.device_ptr(stream),
        }
    }

    pub fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a Arc<CudaStream>,
    ) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
        match &mut self.inner {
            SliceInner::BF16(v) => v.device_ptr_mut(stream),
            SliceInner::F16(v) => v.device_ptr_mut(stream),
            SliceInner::F32(v) => v.device_ptr_mut(stream),
            SliceInner::I32(v) => v.device_ptr_mut(stream),
            SliceInner::I64(v) => v.device_ptr_mut(stream),
        }
    }
}

impl Device {
    pub fn new(device_index: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(device_index).w()?;
        let stream = context.default_stream();
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        Ok(Self { context, stream, blas: Arc::new(blas) })
    }

    pub fn cudarc_stream(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.stream
    }

    pub fn blas(&self) -> &cudarc::cublas::CudaBlas {
        self.blas.as_ref()
    }

    pub fn compile_cu(&self, cu_code: &str, func_name: &str) -> Result<CudaFunction> {
        let opts =
            cudarc::nvrtc::CompileOptions { use_fast_math: Some(true), ..Default::default() };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cu_code, opts).w()?;
        let mdl = self.context.load_module(ptx).w()?;
        let func = mdl.load_function(func_name).w()?;
        Ok(func)
    }

    pub fn compile_ptx(&self, ptx_code: &str, func_name: &'static str) -> Result<CudaFunction> {
        let ptx = cudarc::nvrtc::safe::Ptx::from_src(ptx_code);
        let mdl = self.context.load_module(ptx).w()?;
        let func = mdl.load_function(func_name).w()?;
        Ok(func)
    }

    pub fn zeros(&self, len: usize) -> Result<Slice> {
        let slice = self.stream.alloc_zeros::<f32>(len).w()?;
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

    #[allow(clippy::missing_transmute_annotations)]
    unsafe fn allocate_uninit(&self, dtype: ug::DType, len: usize) -> Result<Self::Slice> {
        let inner = match dtype {
            ug::DType::F32 => {
                let slice = self.stream.alloc::<f32>(len).w()?;
                SliceInner::F32(slice)
            }
            ug::DType::F16 => {
                let slice = self.stream.alloc::<half::f16>(len).w()?;
                SliceInner::F16(slice)
            }
            ug::DType::BF16 => {
                let slice = self.stream.alloc::<half::bf16>(len).w()?;
                SliceInner::BF16(slice)
            }
            ug::DType::I32 => {
                let slice = self.stream.alloc::<i32>(len).w()?;
                SliceInner::I32(slice)
            }
            ug::DType::I64 => {
                let slice = self.stream.alloc::<i64>(len).w()?;
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
        let cu_code = String::from_utf8(cu_code)?;
        let cfg = kernel.launch_config();
        let cfg = LaunchConfig {
            grid_dim: (cfg.grid_dim, 1, 1),
            block_dim: (cfg.block_dim, 1, 1),
            shared_mem_bytes: cfg.shared_mem,
        };
        let func = self.compile_cu(&cu_code, func_name)?;
        Ok(Func { func, stream: self.stream.clone(), cfg })
    }

    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()> {
        use cudarc::driver::PushKernelArg;
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
                let mut builder = f.builder();
                let (a1, _guard1) = a1.device_ptr_mut(&f.stream);
                builder.arg(&a1);
                unsafe { builder.launch(f.cfg).w()? };
            }
            [a1, a2] => {
                let mut builder = f.builder();
                let (a1, _guard1) = a1.device_ptr_mut(&f.stream);
                builder.arg(&a1);
                let (a2, _guard2) = a2.device_ptr_mut(&f.stream);
                builder.arg(&a2);
                unsafe { builder.launch(f.cfg).w()? };
            }
            [a1, a2, a3] => {
                let mut builder = f.builder();
                let (a1, _guard1) = a1.device_ptr_mut(&f.stream);
                builder.arg(&a1);
                let (a2, _guard2) = a2.device_ptr_mut(&f.stream);
                builder.arg(&a2);
                let (a3, _guard3) = a3.device_ptr_mut(&f.stream);
                builder.arg(&a3);
                unsafe { builder.launch(f.cfg).w()? };
            }
            [a1, a2, a3, a4] => {
                let mut builder = f.builder();
                let (a1, _guard1) = a1.device_ptr_mut(&f.stream);
                builder.arg(&a1);
                let (a2, _guard2) = a2.device_ptr_mut(&f.stream);
                builder.arg(&a2);
                let (a3, _guard3) = a3.device_ptr_mut(&f.stream);
                builder.arg(&a3);
                let (a4, _guard4) = a4.device_ptr_mut(&f.stream);
                builder.arg(&a4);
                unsafe { builder.launch(f.cfg).w()? };
            }
            [a1, a2, a3, a4, a5] => {
                let mut builder = f.builder();
                let (a1, _guard1) = a1.device_ptr_mut(&f.stream);
                builder.arg(&a1);
                let (a2, _guard2) = a2.device_ptr_mut(&f.stream);
                builder.arg(&a2);
                let (a3, _guard3) = a3.device_ptr_mut(&f.stream);
                builder.arg(&a3);
                let (a4, _guard4) = a4.device_ptr_mut(&f.stream);
                builder.arg(&a4);
                let (a5, _guard5) = a5.device_ptr_mut(&f.stream);
                builder.arg(&a5);
                unsafe { builder.launch(f.cfg).w()? };
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
            (S::BF16(dst), C::BF16(src)) => self.device.stream.memcpy_htod(src, dst).w()?,
            (S::F16(dst), C::F16(src)) => self.device.stream.memcpy_htod(src, dst).w()?,
            (S::F32(dst), C::F32(src)) => self.device.stream.memcpy_htod(src, dst).w()?,
            (S::I32(dst), C::I32(src)) => self.device.stream.memcpy_htod(src, dst).w()?,
            (S::I64(dst), C::I64(src)) => self.device.stream.memcpy_htod(src, dst).w()?,
            (_, _) => ug::bail!("htod dtype mismatch, dst {:?}, src {:?}", self.dtype(), DT::DTYPE),
        }
        Ok(())
    }

    fn copy_device_to_host<DT: WithDType>(&self, dst: &mut [DT]) -> Result<()> {
        use ug::CpuStorageRefMut as C;
        use SliceInner as S;
        match (&self.inner, DT::to_cpu_storage_mut(dst)) {
            (S::BF16(src), C::BF16(dst)) => self.device.stream.memcpy_dtoh(src, dst).w()?,
            (S::F16(src), C::F16(dst)) => self.device.stream.memcpy_dtoh(src, dst).w()?,
            (S::F32(src), C::F32(dst)) => self.device.stream.memcpy_dtoh(src, dst).w()?,
            (S::I32(src), C::I32(dst)) => self.device.stream.memcpy_dtoh(src, dst).w()?,
            (S::I64(src), C::I64(dst)) => self.device.stream.memcpy_dtoh(src, dst).w()?,
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
