use pyo3::prelude::*;
use std::sync::Arc;
use ug::Slice as S;
use ug_cuda::runtime as cuda;

const MODULE_NAME: &str = "ug-pyo3-mod";

fn w<E: ToString>(err: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
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
    #[pyo3(signature = (s1, s2, s3))]
    fn launch3(&self, s1: &Slice, s2: &Slice, s3: &mut Slice) -> PyResult<()> {
        let len = s3.0.len();
        let len1 = s1.0.len();
        let len2 = s2.0.len();
        if len1 != len {
            py_bail!("length mismatch {len1} <> {len}")
        }
        if len2 != len {
            py_bail!("length mismatch {len2} <> {len}")
        }
        unsafe {
            self.0
                .launch3((
                    s1.0.slice::<f32>().map_err(w)?,
                    s2.0.slice::<f32>().map_err(w)?,
                    s3.0.slice::<f32>().map_err(w)?,
                ))
                .map_err(w)?
        };
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
struct DType(ug::lang::DType);

#[pymethods]
impl DType {
    #[classattr]
    fn f32() -> Self {
        Self(ug::lang::DType::F32)
    }
    #[classattr]
    fn i64() -> Self {
        Self(ug::lang::DType::I64)
    }
    #[classattr]
    fn i32() -> Self {
        Self(ug::lang::DType::I32)
    }
    #[classattr]
    fn bf16() -> Self {
        Self(ug::lang::DType::BF16)
    }
    #[classattr]
    fn f16() -> Self {
        Self(ug::lang::DType::F16)
    }
}

mod op {
    use super::w;
    use pyo3::prelude::*;
    use ug::lang::op;

    #[pyclass]
    #[derive(Clone)]
    pub struct Arg(ug::lang::Arg);

    #[pymethods]
    impl Arg {
        #[staticmethod]
        fn ptr_bf16() -> Self {
            Self(ug::lang::Arg::ptr(ug::DType::BF16))
        }

        #[staticmethod]
        fn ptr_f16() -> Self {
            Self(ug::lang::Arg::ptr(ug::DType::F16))
        }

        #[staticmethod]
        fn ptr_f32() -> Self {
            Self(ug::lang::Arg::ptr(ug::DType::F32))
        }

        #[staticmethod]
        fn ptr_i32() -> Self {
            Self(ug::lang::Arg::ptr(ug::DType::I32))
        }

        #[staticmethod]
        fn f32() -> Self {
            Self(ug::lang::Arg::value(ug::lang::DType::F32))
        }

        #[staticmethod]
        fn i32() -> Self {
            Self(ug::lang::Arg::value(ug::lang::DType::I32))
        }

        fn __str__(&self) -> String {
            format!("{:?}", self.0)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct Ast(op::Ast);

    #[pymethods]
    impl Ast {
        fn __add__(&self, rhs: &Self) -> PyResult<Self> {
            let ast = op::binary(op::BinaryOp::Add, self.0.clone(), rhs.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn __sub__(&self, rhs: &Self) -> PyResult<Self> {
            let ast = op::binary(op::BinaryOp::Sub, self.0.clone(), rhs.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn __mul__(&self, rhs: &Self) -> PyResult<Self> {
            let ast = op::binary(op::BinaryOp::Mul, self.0.clone(), rhs.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn __truediv__(&self, rhs: &Self) -> PyResult<Self> {
            let ast = op::binary(op::BinaryOp::Div, self.0.clone(), rhs.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn __neg__(&self) -> PyResult<Self> {
            let ast = op::unary(op::UnaryOp::Neg, self.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn broadcast(&self, shape: Vec<usize>) -> PyResult<Self> {
            let ast = op::broadcast(self.0.clone(), shape).map_err(w)?;
            Ok(Self(ast))
        }

        fn exp(&self) -> PyResult<Self> {
            let ast = op::unary(op::UnaryOp::Exp, self.0.clone()).map_err(w)?;
            Ok(Self(ast))
        }

        fn sum(&self, axis: usize) -> PyResult<Self> {
            let ast = op::reduce(op::ReduceOp::Sum, self.0.clone(), axis).map_err(w)?;
            Ok(Self(ast))
        }

        fn min(&self, axis: usize) -> PyResult<Self> {
            let ast = op::reduce(op::ReduceOp::Min, self.0.clone(), axis).map_err(w)?;
            Ok(Self(ast))
        }

        fn max(&self, axis: usize) -> PyResult<Self> {
            let ast = op::reduce(op::ReduceOp::Max, self.0.clone(), axis).map_err(w)?;
            Ok(Self(ast))
        }

        fn shape(&self, py: Python) -> PyObject {
            let shape = self.0.shape().dims().to_vec();
            pyo3::types::PyTuple::new_bound(py, shape).into_py(py)
        }
    }

    #[pyfunction]
    pub fn load(arg: Arg, shape: Vec<usize>) -> PyResult<Ast> {
        let layout = op::Layout::from_shape(shape);
        let dtype = match arg.0.type_() {
            ug::lang::Type::Ptr(v) => v,
            ug::lang::Type::Value(_) => py_bail!("unexpected dtype for load {:?}", arg.0),
        };
        let st = op::load(arg.0.id(), layout, dtype).map_err(w)?;
        Ok(Ast(st))
    }

    #[pyfunction]
    pub fn i32(v: i32) -> PyResult<Ast> {
        Ok(Ast(op::cst(v).map_err(w)?))
    }

    #[pyfunction]
    pub fn f32(v: f32) -> PyResult<Ast> {
        Ok(Ast(op::cst(v).map_err(w)?))
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct Store(op::Store);

    #[pymethods]
    impl Store {
        #[new]
        fn new(dst: Arg, shape: Vec<usize>, value: Ast) -> PyResult<Self> {
            let layout = op::Layout::from_shape(shape);
            let st = op::store(dst.0.id(), layout, value.0.clone()).map_err(w)?;
            Ok(Self(st))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct Kernel(op::Kernel);

    #[pymethods]
    impl Kernel {
        #[new]
        fn new(name: String, args: Vec<Arg>, ops: Vec<Store>) -> Self {
            let args = args.into_iter().map(|v| v.0).collect();
            let ops = ops.into_iter().map(|v| v.0).collect();
            Self(op::Kernel::new(name, args, ops))
        }

        fn __str__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn lower(&self) -> PyResult<super::ssa::Kernel> {
            let ssa = self.0.clone().lower(&Default::default()).map_err(w)?;
            Ok(super::ssa::Kernel(ssa))
        }
    }
}

mod ssa {
    use super::{w, DType};
    use pyo3::prelude::*;
    use ug::lang::ssa;

    fn v(id: usize) -> ssa::VarId {
        ssa::VarId::new(id)
    }

    fn a(id: usize) -> ssa::A {
        ssa::A::Var(v(id))
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct Instr(ssa::Instr);

    #[pymethods]
    impl Instr {
        fn __str__(&self) -> String {
            format!("{:?}", self.0)
        }

        #[staticmethod]
        fn const_i32(v: i32) -> Self {
            Self(ssa::Instr::Const(ssa::Const::I32(v)))
        }

        #[staticmethod]
        fn const_f32(v: f32) -> PyResult<Self> {
            Ok(Self(ssa::Instr::Const(v.try_into().map_err(w)?)))
        }

        #[staticmethod]
        fn define_acc_i32(v: i32) -> Self {
            Self(ssa::Instr::DefineAcc(ssa::Const::I32(v)))
        }

        #[staticmethod]
        fn define_acc_f32(v: f32) -> PyResult<Self> {
            Ok(Self(ssa::Instr::Const(v.try_into().map_err(w)?)))
        }

        #[staticmethod]
        fn end_range(start_idx: usize) -> Self {
            Self(ssa::Instr::EndRange { start_idx: ssa::VarId::new(start_idx) })
        }

        #[staticmethod]
        fn range(lo: usize, up: usize, end_idx: usize) -> Self {
            let end_idx = ssa::VarId::new(end_idx);
            Self(ssa::Instr::Range { lo: a(lo), up: a(up), end_idx, step: 1 })
        }

        #[staticmethod]
        fn load(src: usize, offset: usize, dtype: DType) -> Self {
            Self(ssa::Instr::Load { src: v(src), offset: a(offset), dtype: dtype.0 })
        }

        #[staticmethod]
        fn store(dst: usize, offset: usize, value: usize, dtype: DType) -> Self {
            Self(ssa::Instr::Store {
                dst: v(dst),
                offset: a(offset),
                value: a(value),
                dtype: dtype.0,
            })
        }

        #[staticmethod]
        fn define_global(index: usize, dtype: DType) -> Self {
            Self(ssa::Instr::DefineGlobal { index, dtype: dtype.0 })
        }

        #[staticmethod]
        fn special_ti() -> Self {
            Self(ssa::Instr::Special(ssa::Special::ThreadIdx))
        }

        #[staticmethod]
        fn special_bi() -> Self {
            Self(ssa::Instr::Special(ssa::Special::BlockIdx))
        }

        #[staticmethod]
        fn unary(op: &str, arg: usize, dtype: DType) -> PyResult<Self> {
            let op = match op {
                "neg" => ssa::UnaryOp::Neg,
                "exp" => ssa::UnaryOp::Exp,
                _ => py_bail!("unknown unary op '{op}'"),
            };
            Ok(Self(ssa::Instr::Unary { op, arg: a(arg), dtype: dtype.0 }))
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
            Ok(Self(ssa::Instr::Binary { op, lhs: a(lhs), rhs: a(rhs), dtype: dtype.0 }))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    pub struct Kernel(pub ssa::Kernel);

    #[pymethods]
    impl Kernel {
        #[new]
        fn new(instrs: Vec<Instr>) -> PyResult<Self> {
            let instrs = instrs.into_iter().map(|v| v.0).collect();
            Ok(Self(ssa::Kernel::from_instrs(instrs).map_err(w)?))
        }

        fn __str__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn flops_and_mem(&self) -> PyResult<(usize, usize)> {
            let fm = self.0.flops_mem_per_thread().map_err(w)?;
            Ok((fm.flops, fm.mem_in_bytes))
        }

        fn to_list(&self) -> Vec<Instr> {
            self.0.instrs().iter().map(|v| Instr(v.clone())).collect()
        }

        fn cuda_code(&self, name: &str) -> PyResult<String> {
            let mut buf = vec![];
            ug_cuda::code_gen::gen(&mut buf, name, &self.0).map_err(w)?;
            let cuda_code = String::from_utf8(buf)?;
            Ok(cuda_code)
        }
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

    #[pyo3(signature = (ptx_code, func_name, block_dim=1, grid_dim=1, shared_mem_bytes=0))]
    fn compile_ptx(
        &self,
        ptx_code: &str,
        func_name: &str,
        block_dim: u32,
        grid_dim: u32,
        shared_mem_bytes: u32,
    ) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let cfg = cuda::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes,
        };
        let func = self.0.compile_ptx(ptx_code, MODULE_NAME, func_name).map_err(w)?;
        let func = ug_cuda::runtime::Func::new(func, cfg);
        Ok(Func(func))
    }

    #[pyo3(signature = (cu_code, func_name, block_dim=1, grid_dim=1, shared_mem_bytes=0))]
    fn compile_cu(
        &self,
        cu_code: &str,
        func_name: &str,
        block_dim: u32,
        grid_dim: u32,
        shared_mem_bytes: u32,
    ) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let cfg = cuda::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes,
        };
        let func = self.0.compile_cu(cu_code, MODULE_NAME, func_name).map_err(w)?;
        let func = ug_cuda::runtime::Func::new(func, cfg);
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

#[pyclass]
#[derive(Clone)]
struct Arg(ug::lang::Arg);

#[pymethods]
impl Arg {
    #[staticmethod]
    fn ptr_f32() -> Self {
        Self(ug::lang::Arg::ptr(ug::DType::F32))
    }

    #[staticmethod]
    fn ptr_i32() -> Self {
        Self(ug::lang::Arg::ptr(ug::DType::I32))
    }

    #[staticmethod]
    fn i32() -> Self {
        Self(ug::lang::Arg::value(ug::DType::I32))
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass]
#[derive(Clone)]
struct Expr(ug::lang::ExprNode);

#[pymethods]
impl Expr {
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __add__(&self, rhs: &Self) -> Self {
        Self(ug::lang::ExprNode::add(&self.0, &rhs.0))
    }

    fn __mul__(&self, rhs: &Self) -> Self {
        Self(ug::lang::ExprNode::mul(&self.0, &rhs.0))
    }

    #[staticmethod]
    fn load(ptr: &Arg, offset: &IndexExpr, len: &IndexExpr, stride: &IndexExpr) -> PyResult<Self> {
        let load = ug::lang::ExprNode::load(&ptr.0, &offset.0, &len.0, &stride.0).map_err(w)?;
        Ok(Self(load))
    }
}

#[pyclass]
#[derive(Clone)]
struct IndexExpr(ug::lang::IndexExprNode);

#[pymethods]
impl IndexExpr {
    #[staticmethod]
    fn program_id() -> Self {
        Self(ug::lang::IndexExprNode::program_id())
    }

    #[staticmethod]
    fn cst(v: usize) -> Self {
        Self(ug::lang::IndexExprNode::cst(v))
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __add__(&self, rhs: &Self) -> Self {
        Self(ug::lang::IndexExprNode::add(&self.0, &rhs.0))
    }

    fn __mul__(&self, rhs: &Self) -> Self {
        Self(ug::lang::IndexExprNode::mul(&self.0, &rhs.0))
    }
}

#[pyclass]
#[derive(Clone)]
struct Ops(ug::lang::Ops);

#[pymethods]
impl Ops {
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    #[staticmethod]
    fn store(
        dst: &Arg,
        offset: &IndexExpr,
        len: &IndexExpr,
        stride: &IndexExpr,
        value: &Expr,
    ) -> Self {
        Self(ug::lang::Ops::store(&dst.0, &offset.0, &len.0, &stride.0, &value.0))
    }
}

#[pyclass]
#[derive(Clone)]
struct Kernel(ug::lang::Kernel);

#[pymethods]
impl Kernel {
    #[new]
    fn new(name: String, args: Vec<Arg>, ops: Vec<Ops>) -> Self {
        let args = args.into_iter().map(|v| v.0).collect();
        let ops = ops.into_iter().map(|v| v.0).collect();
        Self(ug::lang::Kernel::new(name, args, ops))
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn lower(&self) -> PyResult<ssa::Kernel> {
        let ssa = self.0.lower().map_err(w)?;
        Ok(ssa::Kernel(ssa))
    }
}

#[pymodule]
#[pyo3(name = "ug")]
fn mod_(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    let ssa = PyModule::new_bound(py, "ssa")?;
    ssa.add_class::<ssa::Kernel>()?;
    ssa.add_class::<ssa::Instr>()?;

    let lang = PyModule::new_bound(py, "lang")?;
    lang.add_class::<Kernel>()?;
    lang.add_class::<Ops>()?;
    lang.add_class::<Arg>()?;
    lang.add_class::<Expr>()?;
    lang.add_class::<IndexExpr>()?;

    let op = PyModule::new_bound(py, "op")?;
    op.add_class::<op::Arg>()?;
    op.add_class::<op::Ast>()?;
    op.add_class::<op::Kernel>()?;
    op.add_class::<op::Store>()?;
    op.add_function(wrap_pyfunction!(op::i32, m)?)?;
    op.add_function(wrap_pyfunction!(op::f32, m)?)?;
    op.add_function(wrap_pyfunction!(op::load, m)?)?;

    m.add_class::<Device>()?;
    m.add_class::<DType>()?;
    m.add_class::<Func>()?;
    m.add_class::<Slice>()?;
    m.add_submodule(&ssa)?;
    m.add_submodule(&lang)?;
    m.add_submodule(&op)?;
    Ok(())
}
