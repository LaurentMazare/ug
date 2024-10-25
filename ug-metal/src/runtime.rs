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

    pub fn zeros<T>(&self, len: usize) -> Result<Slice<T>> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (len * std::mem::size_of::<T>()) as u64;
        let buffer = self.device.new_buffer(bytes_len, options);
        Ok(Slice { buffer, _phantom: std::marker::PhantomData, len })
    }

    pub fn slice_from_values<T>(&self, data: &[T]) -> Result<Slice<T>> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let len = data.len();
        let bytes_len = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(ptr, bytes_len, options);
        Ok(Slice { buffer, _phantom: std::marker::PhantomData, len })
    }
}

pub struct Slice<T> {
    buffer: metal::Buffer,
    _phantom: std::marker::PhantomData<T>,
    len: usize,
}

impl<T: Clone> Slice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        assert!(!ptr.is_null());
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len()) };
        slice.to_vec()
    }

    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}
