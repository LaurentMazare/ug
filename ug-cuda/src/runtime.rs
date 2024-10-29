pub use cudarc::driver::LaunchConfig;
use std::sync::Arc;
use ug::{Error, Result};

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

#[derive(Clone)]
pub struct Slice {
    // TODO(laurent): handle some general types.
    slice: cudarc::driver::CudaSlice<f32>,
    len: usize,
}

impl Slice {
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let vec = self.slice.device().dtoh_sync_copy(&self.slice).w()?;
        Ok(vec)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slice(&self) -> &cudarc::driver::CudaSlice<f32> {
        &self.slice
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
        Ok(Slice { slice, len })
    }

    pub fn slice_from_values(&self, vs: &[f32]) -> Result<Slice> {
        let len = vs.len();
        let slice = self.device.htod_sync_copy(vs).w()?;
        Ok(Slice { slice, len })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize().w()?;
        Ok(())
    }
}

impl ug::Device for Device {
    type Slice<DT: ug::WithDType> = cudarc::driver::CudaSlice<DT>;

    #[allow(clippy::missing_transmute_annotations)]
    fn allocate_uninit<D: ug::WithDType>(&self, len: usize) -> Result<Self::Slice<D>> {
        let type_id = std::any::TypeId::of::<D>();
        let slice = match D::DTYPE {
            ug::DType::F32 => {
                // This assert + transmute is ugly but avoids adding the DeviceRepr constraint in
                // WithDType.
                assert_eq!(type_id, std::any::TypeId::of::<f32>());
                unsafe { std::mem::transmute(self.device.alloc::<f32>(len).w()?) }
            }
            ug::DType::F16 => {
                assert_eq!(type_id, std::any::TypeId::of::<half::f16>());
                unsafe { std::mem::transmute(self.device.alloc::<half::f16>(len).w()?) }
            }
            ug::DType::BF16 => {
                assert_eq!(type_id, std::any::TypeId::of::<half::bf16>());
                unsafe { std::mem::transmute(self.device.alloc::<half::bf16>(len).w()?) }
            }
            ug::DType::I32 => {
                assert_eq!(type_id, std::any::TypeId::of::<i32>());
                unsafe { std::mem::transmute(self.device.alloc::<i32>(len).w()?) }
            }
            ug::DType::I64 => {
                assert_eq!(type_id, std::any::TypeId::of::<i64>());
                unsafe { std::mem::transmute(self.device.alloc::<i64>(len).w()?) }
            }
        };
        Ok(slice)
    }

    fn copy_host_to_device<D: ug::WithDType>(_src: &[D], _dst: &mut Self::Slice<D>) -> Result<()> {
        todo!()
    }

    fn copy_device_to_host<D: ug::WithDType>(_src: &Self::Slice<D>, _dst: &mut [D]) -> Result<()> {
        todo!()
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
