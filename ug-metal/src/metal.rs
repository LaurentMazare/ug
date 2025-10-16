use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType,
    MTLDevice, MTLFunction, MTLFunctionConstantValues, MTLLibrary, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};
use std::{borrow::Cow, ffi::c_void, ptr};

type Error = ug::Error;
type Result<T> = ug::Result<T>;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Buffer { raw }
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

pub type MetalResource = ProtocolObject<dyn MTLResource>;
impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> ComputePipeline {
        ComputePipeline { raw }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    encoding_ended: bool,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder { raw, encoding_ended: false }
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe { self.raw.setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index) }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&mut self) {
        use objc2_metal::MTLCommandEncoder as _;
        if !self.encoding_ended {
            self.raw.endEncoding();
            self.encoding_ended = true;
        }
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        self.raw.setComputePipelineState(pipeline.as_ref());
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

pub fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer> {
    command_queue
        .commandBuffer()
        .map(CommandBuffer::new)
        .ok_or(Error::Msg("Could not create command buffer".to_string()))
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { raw }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref().computeCommandEncoder().map(ComputeCommandEncoder::new).unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted()
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}

pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl AsRef<ProtocolObject<dyn MTLDevice>> for Device {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

impl Device {
    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    pub fn all() -> Vec<Self> {
        MTLCreateSystemDefaultDevice().into_iter().map(|raw| Device { raw }).collect()
    }

    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice().map(|raw| Device { raw })
    }

    pub fn new_buffer(&self, length: usize, options: MTLResourceOptions) -> Result<Buffer> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(Buffer::new)
            .ok_or(Error::Msg("Could not allocate metal buffer".to_string()))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void).unwrap();
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(Buffer::new)
                .ok_or(Error::Msg("Could not allocate metal buffer".to_string()))
        }
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library> {
        let raw = self
            .as_ref()
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            .unwrap();

        Ok(Library::new(raw))
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline> {
        self.as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map(|raw| ComputePipeline::new(raw))
            .map_err(|err| Error::Msg(format!("Could not create compute pipeline state: {}", err)))
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue> {
        self.as_ref()
            .newCommandQueue()
            .ok_or(Error::Msg("Could not create command queue".to_string()))
    }
}

#[derive(Clone)]
pub struct Function {
    raw: Retained<ProtocolObject<dyn MTLFunction>>,
}

impl AsRef<ProtocolObject<dyn MTLFunction>> for Function {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLFunction> {
        &self.raw
    }
}

pub struct FunctionConstantValues {
    raw: Retained<MTLFunctionConstantValues>,
}

impl FunctionConstantValues {
    pub fn new() -> FunctionConstantValues {
        FunctionConstantValues { raw: MTLFunctionConstantValues::new() }
    }

    pub fn set_constant_value_at_index<T>(&self, value: &T, dtype: MTLDataType, index: usize) {
        let value = ptr::NonNull::new(value as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setConstantValue_type_atIndex(value, dtype, index) }
    }
}

impl Default for FunctionConstantValues {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLLibrary>>) -> Library {
        Library { raw }
    }

    pub fn get_function(
        &self,
        name: &str,
        constant_values: Option<&ConstantValues>,
    ) -> Result<Function> {
        let function = match constant_values {
            Some(constant_values) => self
                .raw
                .newFunctionWithName_constantValues_error(
                    &NSString::from_str(name),
                    &constant_values.function_constant_values().raw,
                )
                .map_err(|e| Error::Msg(format!("Failed to load function '{}': {}", name, e)))?,
            None => self
                .raw
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Msg(format!("Failed to load function '{}'", name)))?,
        };

        Ok(Function { raw: function })
    }
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            // usize is usually u64 aka ulong, but can be u32 on 32-bit systems.
            // https://developer.apple.com/documentation/objectivec/nsuinteger
            Value::USize(_) => MTLDataType::ULong,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(v, ty, *index);
                }
            }
        }
        f
    }
}
