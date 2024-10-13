use pyo3::prelude::*;
use std::sync::Arc;
use ug::lang::ssa;
use ug_cuda::runtime as cuda;

const MODULE_NAME: &str = "ug-pyo3-mod";

fn w<E: ToString>(err: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn v(id: usize) -> ssa::VarId {
    ssa::VarId::new(id)
}

#[macro_export]
macro_rules! py_bail {
    ($msg:literal $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($msg)))
    };
    ($err:expr $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($err)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($fmt, $($arg)*)))
    };
}

#[pyclass]
struct Func(cuda::Func);

#[pymethods]
impl Func {
    #[pyo3(signature = (s1, s2, s3, block_dim=1, grid_dim=1, shared_mem_bytes=0))]
    fn launch3(
        &self,
        s1: &Slice,
        s2: &Slice,
        s3: &mut Slice,
        block_dim: u32,
        grid_dim: u32,
        shared_mem_bytes: u32,
    ) -> PyResult<()> {
        let len = s3.0.len();
        let len1 = s1.0.len();
        let len2 = s2.0.len();
        if len1 != len {
            py_bail!("length mismatch {len1} <> {len}")
        }
        if len2 != len {
            py_bail!("length mismatch {len2} <> {len}")
        }
        let cfg = cuda::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes,
        };
        unsafe { self.0.launch3(s1.0.slice(), s2.0.slice(), s3.0.slice(), cfg).map_err(w)? };
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
struct Slice(Arc<cuda::Slice>);

#[pymethods]
impl Slice {
    fn to_vec(&self) -> PyResult<Vec<f32>> {
        let v = self.0.to_vec().map_err(w)?;
        Ok(v)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[pyclass]
#[derive(Clone)]
struct DType(ssa::DType);

#[pymethods]
impl DType {
    #[classattr]
    fn f32() -> Self {
        Self(ssa::DType::F32)
    }
    #[classattr]
    fn i32() -> Self {
        Self(ssa::DType::I32)
    }
    #[classattr]
    fn ptr_f32() -> Self {
        Self(ssa::DType::PtrF32)
    }
    #[classattr]
    fn ptr_i32() -> Self {
        Self(ssa::DType::PtrI32)
    }
}

#[pyclass]
#[derive(Clone)]
struct Instr(ssa::Instr);

#[pymethods]
impl Instr {
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    #[staticmethod]
    fn const_i32(v: i32) -> Self {
        Instr(ssa::Instr::Const(ssa::Const::I32(v)))
    }

    #[staticmethod]
    fn const_f32(v: f32) -> Self {
        Instr(ssa::Instr::Const(ssa::Const::F32(v)))
    }

    #[staticmethod]
    fn define_acc_i32(v: i32) -> Self {
        Instr(ssa::Instr::DefineAcc(ssa::Const::I32(v)))
    }

    #[staticmethod]
    fn define_acc_f32(v: f32) -> Self {
        Instr(ssa::Instr::DefineAcc(ssa::Const::F32(v)))
    }

    #[staticmethod]
    fn end_range(start_idx: usize) -> Self {
        Instr(ssa::Instr::EndRange { start_idx })
    }

    #[staticmethod]
    fn range(lo: usize, up: usize, end_idx: usize) -> Self {
        Instr(ssa::Instr::Range { lo: v(lo), up: v(up), end_idx })
    }

    #[staticmethod]
    fn load(src: usize, offset: usize, dtype: DType) -> Self {
        Instr(ssa::Instr::Load { src: v(src), offset: v(offset), dtype: dtype.0 })
    }

    #[staticmethod]
    fn store(dst: usize, offset: usize, value: usize, dtype: DType) -> Self {
        Instr(ssa::Instr::Store { dst: v(dst), offset: v(offset), value: v(value), dtype: dtype.0 })
    }

    #[staticmethod]
    fn define_global(index: usize, dtype: DType) -> Self {
        Instr(ssa::Instr::DefineGlobal { index, dtype: dtype.0 })
    }

    #[staticmethod]
    fn special_l() -> Self {
        Instr(ssa::Instr::Special(ssa::Special::LocalIdx))
    }

    #[staticmethod]
    fn special_g() -> Self {
        Instr(ssa::Instr::Special(ssa::Special::GridIdx))
    }

    #[staticmethod]
    fn unary(op: &str, arg: usize, dtype: DType) -> PyResult<Self> {
        let op = match op {
            "neg" => ssa::UnaryOp::Neg,
            "exp" => ssa::UnaryOp::Exp,
            _ => py_bail!("unknown unary op '{op}'"),
        };
        Ok(Instr(ssa::Instr::Unary { op, arg: v(arg), dtype: dtype.0 }))
    }

    #[staticmethod]
    fn binary(op: &str, lhs: usize, rhs: usize, dtype: DType) -> PyResult<Self> {
        let op = match op {
            "add" => ssa::BinaryOp::Add,
            "mul" => ssa::BinaryOp::Mul,
            "sub" => ssa::BinaryOp::Sub,
            "div" => ssa::BinaryOp::Div,
            _ => py_bail!("unknown binary op '{op}'"),
        };
        Ok(Instr(ssa::Instr::Binary { op, lhs: v(lhs), rhs: v(rhs), dtype: dtype.0 }))
    }
}

#[pyclass]
#[derive(Clone)]
struct Kernel(ssa::Kernel);

#[pymethods]
impl Kernel {
    #[new]
    fn new(instrs: Vec<Instr>) -> Self {
        let instrs = instrs.into_iter().map(|v| v.0).collect();
        Self(ssa::Kernel { instrs })
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn flops_and_mem(&self) -> PyResult<(usize, usize)> {
        let fm = self.0.flops_mem_per_thread().map_err(w)?;
        Ok((fm.flops, fm.mem_in_bytes))
    }

    fn to_list(&self) -> Vec<Instr> {
        self.0.instrs.iter().map(|v| Instr(v.clone())).collect()
    }

    fn cuda_code(&self, name: &str) -> PyResult<String> {
        let mut buf = vec![];
        ug_cuda::code_gen::gen(&mut buf, name, &self.0).map_err(w)?;
        let cuda_code = String::from_utf8(buf)?;
        Ok(cuda_code)
    }
}

#[pyclass]
struct Device(cuda::Device);

#[pymethods]
impl Device {
    #[new]
    #[pyo3(signature = (device_id))]
    fn new(device_id: usize) -> PyResult<Self> {
        let device = cuda::Device::new(device_id).map_err(w)?;
        Ok(Self(device))
    }

    #[pyo3(signature = (ptx_code, func_name))]
    fn compile_ptx(&self, ptx_code: &str, func_name: &str) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let func = self.0.compile_ptx(ptx_code, MODULE_NAME, func_name).map_err(w)?;
        Ok(Func(func))
    }

    #[pyo3(signature = (cu_code, func_name))]
    fn compile_cu(&self, cu_code: &str, func_name: &str) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let func = self.0.compile_cu(cu_code, MODULE_NAME, func_name).map_err(w)?;
        Ok(Func(func))
    }

    #[pyo3(signature = (len,))]
    fn zeros(&self, len: usize) -> PyResult<Slice> {
        let slice = self.0.zeros(len).map_err(w)?;
        Ok(Slice(Arc::new(slice)))
    }

    #[pyo3(signature = (vs,))]
    fn slice(&self, vs: Vec<f32>) -> PyResult<Slice> {
        let slice = self.0.slice_from_values(&vs).map_err(w)?;
        Ok(Slice(Arc::new(slice)))
    }

    #[pyo3(signature = ())]
    fn synchronize(&self) -> PyResult<()> {
        self.0.synchronize().map_err(w)
    }
}

#[pymodule]
#[pyo3(name = "ug")]
fn mod_(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Func>()?;
    m.add_class::<Kernel>()?;
    m.add_class::<Device>()?;
    m.add_class::<DType>()?;
    m.add_class::<Instr>()?;
    m.add_class::<Slice>()?;
    Ok(())
}
