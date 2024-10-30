pub use cudarc::driver::DeviceSlice;
pub use cudarc::driver::LaunchConfig;
use std::sync::Arc;
use ug::{Device as D, Error, Result, Slice as S, WithDType};

pub trait WithErr {
    type T;
    fn w(self) -> Result<Self::T>;
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

#[derive(Clone)]
pub struct Func {
    func: cudarc::driver::CudaFunction,
}

impl Func {
    /// Launch a kernel with one argument.
    ///
    /// # Safety
    /// Launching a kernel is always unsafe...
    pub unsafe fn launch1<Params: cudarc::driver::DeviceRepr>(
        &self,
        p: Params,
        cfg: LaunchConfig,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;
        let func = self.func.clone();
        unsafe { func.launch(cfg, (p,)).w()? };
        Ok(())
    }

    /// Launch a kernel with 2 arguments.
    ///
    /// # Safety
    /// Launching a kernel is always unsafe...
    pub unsafe fn launch2<Params: cudarc::driver::DeviceRepr>(
        &self,
        p1: Params,
        p2: Params,
        cfg: LaunchConfig,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;
        let func = self.func.clone();
        unsafe { func.launch(cfg, (p1, p2)).w()? };
        Ok(())
    }

    /// Launch a kernel with 3 arguments.
    ///
    /// # Safety
    /// Launching a kernel is always unsafe...
    pub unsafe fn launch3<Params: cudarc::driver::DeviceRepr>(
        &self,
        p1: Params,
        p2: Params,
        p3: Params,
        cfg: LaunchConfig,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;
        let func = self.func.clone();
        unsafe { func.launch(cfg, (p1, p2, p3)).w()? };
        Ok(())
    }
}

#[derive(Clone)]
pub struct Device {
    device: Arc<cudarc::driver::CudaDevice>,
}

// A GADT based solution would seem better than this variant but not sure how to do this in rust.
#[derive(Clone)]
pub enum SliceInner {
    F32(cudarc::driver::CudaSlice<f32>),
    F16(cudarc::driver::CudaSlice<half::f16>),
    BF16(cudarc::driver::CudaSlice<half::bf16>),
    I32(cudarc::driver::CudaSlice<i32>),
    I64(cudarc::driver::CudaSlice<i64>),
}

#[derive(Clone)]
pub struct Slice {
    inner: SliceInner,
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
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        match &self.inner {
            SliceInner::F32(slice) => slice.device().dtoh_sync_copy(slice).w(),
            _ => todo!(),
        }
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            SliceInner::F32(slice) => slice.len(),
            SliceInner::F16(slice) => slice.len(),
            SliceInner::BF16(slice) => slice.len(),
            SliceInner::I32(slice) => slice.len(),
            SliceInner::I64(slice) => slice.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice<D: ToSlice>(&self) -> Result<&cudarc::driver::CudaSlice<D>> {
        ToSlice::slice(self)
    }
}

impl Device {
    pub fn new(device_index: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(device_index).w()?;
        Ok(Self { device })
    }

    pub fn compile_cu(
        &self,
        cu_code: &str,
        module_name: &str,
        func_name: &'static str,
    ) -> Result<Func> {
        let opts =
            cudarc::nvrtc::CompileOptions { use_fast_math: Some(true), ..Default::default() };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cu_code, opts).w()?;
        self.device.load_ptx(ptx, module_name, &[func_name]).w()?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => ug::bail!("unknown function {module_name}::{func_name}"),
        };
        Ok(Func { func })
    }

    pub fn compile_ptx(
        &self,
        ptx_code: &str,
        module_name: &str,
        func_name: &'static str,
    ) -> Result<Func> {
        let ptx = cudarc::nvrtc::safe::Ptx::from_src(ptx_code);
        self.device.load_ptx(ptx, module_name, &[func_name]).w()?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => ug::bail!("unknown function {module_name}::{func_name}"),
        };
        Ok(Func { func })
    }

    pub fn zeros(&self, len: usize) -> Result<Slice> {
        let slice = self.device.alloc_zeros::<f32>(len).w()?;
        Ok(Slice { inner: SliceInner::F32(slice), device: self.clone() })
    }

    pub fn slice_from_values<D: WithDType>(&self, vs: &[D]) -> Result<Slice> {
        let mut slice = unsafe { self.allocate_uninit::<D>(vs.len())? };
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

    #[allow(clippy::missing_transmute_annotations)]
    unsafe fn allocate_uninit<D: WithDType>(&self, len: usize) -> Result<Self::Slice> {
        let inner = match D::DTYPE {
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
        use ug::dtype::CpuStorageRef as C;
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
        use ug::dtype::CpuStorageRefMut as C;
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
}
