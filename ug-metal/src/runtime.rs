use crate::utils::EncoderParam;
use metal::{FunctionConstantValues, MTLDataType};
use std::sync::OnceLock;
use ug::{Error, Result};

const MLX_GEMM: &str = include_str!("mlx_gemm.metal");

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

    pub fn new_command_buffer(&self) -> &metal::CommandBufferRef {
        self.cq.new_command_buffer()
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
            encoder.use_resource(&arg.buffer, metal::MTLResourceUsage::Read);
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
        dst: &mut Self::Slice,
        lhs: &Self::Slice,
        rhs: &Self::Slice,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &ug::Layout,
        rhs_l: &ug::Layout,
    ) -> Result<()> {
        let cb = self.cq.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        call_mlx_gemm(
            &dst.device,
            encoder,
            GemmDType::F32,
            bmnk,
            lhs_l.strides(),
            lhs_l.offset(),
            lhs.buffer(),
            rhs_l.strides(),
            rhs_l.offset(),
            rhs.buffer(),
            dst.buffer(),
        )?;
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
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
            Value::USize(_) => MTLDataType::UInt,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        use std::ffi::c_void;
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(
                        v as *const usize as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(
                        v as *const u16 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[allow(clippy::too_many_arguments)]
fn call_mlx_gemm(
    device: &Device,
    encoder: &metal::ComputeCommandEncoderRef,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &metal::Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &metal::Buffer,
    output: &metal::Buffer,
) -> Result<()> {
    use std::ffi::c_void;
    static LIB: OnceLock<core::result::Result<metal::Library, String>> = OnceLock::new();

    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        ug::bail!("matmul striding error {lhs_stride:?} {rhs_stride:?} {m} {n} {k}")
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        ug::bail!("matmul striding error {lhs_stride:?} {rhs_stride:?} {m} {n} {k}")
    };
    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a =
        if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] } else { m * k };
    let batch_stride_b =
        if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] } else { n * k };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };
    let batch_strides = [gemm_params.batch_stride_a, gemm_params.batch_stride_b];

    // TODO(laurent): generate the name
    // template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]]
    let name = match (dtype, a_trans, b_trans) {
        (GemmDType::F32, false, false) => "gemm_nn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, false) => "gemm_tn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, false, true) => "gemm_nt_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, true) => "gemm_tt_f32_f32_32_32_16_2_2",
        (GemmDType::BF16, false, false) => "gemm_nn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, false) => "gemm_tn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, false, true) => "gemm_nt_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, true) => "gemm_tt_bf16_bf16_32_32_16_2_2",
        (GemmDType::F16, false, false) => "gemm_nn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, false) => "gemm_tn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, false, true) => "gemm_nt_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, true) => "gemm_tt_f16_f16_32_32_16_2_2",
    };

    // TODO: Avoid recompiling the code for each matmul.
    let lib = LIB.get_or_init(|| {
        device.device.new_library_with_source(MLX_GEMM, &metal::CompileOptions::new())
    });
    let lib = match lib {
        Ok(lib) => lib,
        Err(err) => ug::bail!("error compiling the gemm kernels {err}"),
    };
    let func =
        lib.get_function(name, constants.as_ref().map(|c| c.function_constant_values())).w()?;

    let pipeline = device.device.new_compute_pipeline_state_with_function(&func).w()?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as metal::NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as metal::NSUInteger);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        6, // batch_shape
        std::mem::size_of::<i32>() as u64,
        &(b as i32) as *const i32 as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
        batch_strides.as_ptr() as *const c_void,
    );

    let grid_size = metal::MTLSize {
        width: tn as u64,
        height: tm as u64,
        depth: /* batch_size_out */ b as u64,
    };
    let group_size = metal::MTLSize { width: 32, height: wn, depth: wm };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}
