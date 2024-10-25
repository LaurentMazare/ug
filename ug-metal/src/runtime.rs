use std::sync::Arc;
use ug::{Error, Result};

pub trait WithErr {
    type T;
    fn w(self) -> Result<Self::T>;
}

impl<T> WithErr for std::result::Result<T, String> {
    type T = T;
    fn w(self) -> Result<Self::T> {
        self.map_err(|v| Error::Msg(v).bt())
    }
}

#[derive(Clone)]
pub struct Func {
    inner: metal::Function,
    device: Device,
}

impl Func {
    pub fn pipeline(&self) -> Result<metal::ComputePipelineState> {
        let pipeline =
            self.device.device.new_compute_pipeline_state_with_function(&self.inner).w()?;
        Ok(pipeline)
    }
}

#[derive(Clone)]
pub struct Device {
    device: Arc<metal::Device>,
}

impl Device {
    pub fn new() -> Result<Self> {
        let device = match metal::Device::system_default() {
            Some(device) => device,
            None => ug::bail!("no default device found"),
        };
        let device = Arc::new(device);
        Ok(Self { device })
    }

    pub fn compile_metal(&self, metal_code: &str, func_name: &str) -> Result<Func> {
        let lib =
            self.device.new_library_with_source(metal_code, &metal::CompileOptions::new()).w()?;
        let inner = lib.get_function(func_name, None).w()?;
        Ok(Func { inner, device: self.clone() })
    }
}
