use crate::utils::EncoderParam;
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
    inner: metal::Function,
    launch_config: ug::lang::LaunchConfig,
    device: Device,
}

impl Func {
    pub fn pipeline(&self) -> Result<metal::ComputePipelineState> {
        let pipeline =
            self.device.device.new_compute_pipeline_state_with_function(&self.inner).w()?;
        Ok(pipeline)
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    device: metal::Device,
    cq: metal::CommandQueue,
}

impl Device {
    pub fn new_command_queue(&self) -> metal::CommandQueue {
        self.device.new_command_queue()
    }

    pub fn new() -> Result<Self> {
        let device = match metal::Device::system_default() {
            Some(device) => device,
            None => ug::bail!("no default device found"),
        };
        let cq = device.new_command_queue();
        Ok(Self { device, cq })
    }

    pub fn compile_metal(
        &self,
        metal_code: &str,
        func_name: &str,
        launch_config: ug::lang::LaunchConfig,
    ) -> Result<Func> {
        let lib =
            self.device.new_library_with_source(metal_code, &metal::CompileOptions::new()).w()?;
        let inner = lib.get_function(func_name, None).w()?;
        Ok(Func { inner, device: self.clone(), launch_config })
    }

    pub fn zeros<T>(&self, len: usize) -> Result<SliceT<T>> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (len * std::mem::size_of::<T>()) as u64;
        let buffer = self.device.new_buffer(bytes_len, options);
        Ok(SliceT { buffer, _phantom: std::marker::PhantomData, len })
    }

    pub fn slice_from_values<T>(&self, data: &[T]) -> Result<SliceT<T>> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let len = data.len();
        let bytes_len = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(ptr, bytes_len, options);
        Ok(SliceT { buffer, _phantom: std::marker::PhantomData, len })
    }
}

impl ug::Device for Device {
    type Slice = Slice;
    type Func = Func;

    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()> {
        let cb = self.cq.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        let pl = f.pipeline()?;
        encoder.set_compute_pipeline_state(&pl);
        for (index, arg) in args.iter().enumerate() {
            <&metal::Buffer>::set_param(encoder, index as u64, &arg.buffer);
            encoder.use_resource(&arg.buffer, metal::MTLResourceUsage::Write);
        }
        let grid_size = metal::MTLSize::new(f.launch_config.grid_dim as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(f.launch_config.block_dim as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    fn matmul(
        &self,
        _dst: &mut Self::Slice,
        _lhs: &Self::Slice,
        _rhs: &Self::Slice,
        _bmnk: (usize, usize, usize, usize),
        _lhs_l: &ug::Layout,
        _rhs_l: &ug::Layout,
    ) -> Result<()> {
        todo!()
    }

    fn compile(&self, kernel: &ug::lang::ssa::Kernel, name: Option<&str>) -> Result<Self::Func> {
        let mut buf = vec![];
        let kernel_id = KernelId::new().as_usize();
        let name = match name {
            Some(name) => &format!("ug_{name}_{kernel_id}"),
            None => &format!("ug_{kernel_id}"),
        };
        crate::code_gen::gen(&mut buf, name, kernel)?;
        let metal_code = String::from_utf8(buf)?;
        self.compile_metal(&metal_code, name, *kernel.launch_config())
    }

    fn synchronize(&self) -> Result<()> {
        // cb.commit();
        // cb.wait_until_completed();
        todo!()
    }

    unsafe fn allocate_uninit(&self, dtype: ug::DType, len: usize) -> Result<Self::Slice> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (len * dtype.size_in_bytes()) as u64;
        let buffer = self.device.new_buffer(bytes_len, options);
        Ok(Slice { buffer, device: self.clone(), len, dtype })
    }

    fn use_grid() -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct SliceT<T> {
    buffer: metal::Buffer,
    _phantom: std::marker::PhantomData<T>,
    len: usize,
}

impl<T: Clone> SliceT<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_vec(&self) -> Vec<T> {
        // let buffer = self.device.new_buffer_managed(size)?;
        // {
        //     let command_buffer = self.device.command_buffer()?;
        //     command_buffer.set_label("to_cpu");
        //     let blit = command_buffer.new_blit_command_encoder();
        //     blit.set_label("blit_to_cpu");
        //     blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
        //     blit.end_encoding();
        // }
        // self.device.wait_until_completed()?;
        let ptr = self.buffer.contents() as *const T;
        assert!(!ptr.is_null());
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len()) };
        slice.to_vec()
    }

    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Slice {
    buffer: metal::Buffer,
    device: Device,
    dtype: ug::DType,
    len: usize,
}

impl ug::Slice for Slice {
    type Device = Device;

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> ug::DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_vec<DT: ug::WithDType>(&self) -> Result<Vec<DT>> {
        let ptr = self.buffer.contents() as *const DT;
        if ptr.is_null() {
            ug::bail!("unexpected null pointer")
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        Ok(slice.to_vec())
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn copy_host_to_device<DT: ug::WithDType>(&mut self, src: &[DT]) -> Result<()> {
        if self.len != src.len() {
            ug::bail!("size mismatch in copy_host_to_device, src {}, dst: {}", src.len(), self.len)
        }
        let ptr = self.buffer.contents() as *mut DT;
        if ptr.is_null() {
            ug::bail!("unexpected null pointer")
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.len) };
        slice.copy_from_slice(src);
        Ok(())
    }

    fn copy_device_to_host<DT: ug::WithDType>(&self, dst: &mut [DT]) -> Result<()> {
        if self.len != dst.len() {
            ug::bail!("size mismatch in copy_device_to_host, src {}, dst: {}", self.len, dst.len())
        }
        let ptr = self.buffer.contents() as *const DT;
        if ptr.is_null() {
            ug::bail!("unexpected null pointer")
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        dst.copy_from_slice(slice);
        Ok(())
    }
}

impl Slice {
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}
