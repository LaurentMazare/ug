use anyhow::Result;
use std::sync::Arc;

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
        el_count: usize,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        let cfg = cudarc::driver::LaunchConfig::for_num_elems(el_count as u32);
        let func = self.func.clone();
        unsafe { func.launch(cfg, (p,))? };
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
        el_count: usize,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        let cfg = cudarc::driver::LaunchConfig::for_num_elems(el_count as u32);
        let func = self.func.clone();
        unsafe { func.launch(cfg, (p1, p2))? };
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
        el_count: usize,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (el_count as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self.func.clone();
        unsafe { func.launch(cfg, (p1, p2, p3))? };
        Ok(())
    }
}

#[derive(Clone)]
pub struct Device {
    device: Arc<cudarc::driver::CudaDevice>,
}

#[derive(Clone)]
pub struct Slice {
    slice: cudarc::driver::CudaSlice<f32>,
    len: usize,
}

impl Slice {
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let vec = self.slice.device().dtoh_sync_copy(&self.slice)?;
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
        let device = cudarc::driver::CudaDevice::new(device_index)?;
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
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cu_code, opts)?;
        self.device.load_ptx(ptx, module_name, &[func_name])?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => anyhow::bail!("unknown function {module_name}::{func_name}"),
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
        self.device.load_ptx(ptx, module_name, &[func_name])?;
        let func = match self.device.get_func(module_name, func_name) {
            Some(func) => func,
            None => anyhow::bail!("unknown function {module_name}::{func_name}"),
        };
        Ok(Func { func })
    }

    pub fn zeros(&self, len: usize) -> Result<Slice> {
        let slice = self.device.alloc_zeros::<f32>(len)?;
        Ok(Slice { slice, len })
    }

    pub fn slice_from_values(&self, vs: &[f32]) -> Result<Slice> {
        let len = vs.len();
        let slice = self.device.htod_sync_copy(vs)?;
        Ok(Slice { slice, len })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()?;
        Ok(())
    }
}
