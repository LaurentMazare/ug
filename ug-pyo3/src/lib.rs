use pyo3::prelude::*;
use std::sync::Arc;
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
        unsafe { self.0.launch3(s1.0.slice(), s2.0.slice(), s3.0.slice(), len).map_err(w)? };
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
struct Device(cuda::Device);

#[pymethods]
impl Device {
    #[new]
    fn new(device_id: usize) -> PyResult<Self> {
        let device = cuda::Device::new(device_id).map_err(w)?;
        Ok(Self(device))
    }

    fn compile_ptx(&self, ptx_code: &str, func_name: &str) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let func = self.0.compile_ptx(ptx_code, MODULE_NAME, func_name).map_err(w)?;
        Ok(Func(func))
    }

    fn compile_cu(&self, cu_code: &str, func_name: &str) -> PyResult<Func> {
        let func_name = Box::leak(func_name.to_string().into_boxed_str());
        let func = self.0.compile_cu(cu_code, MODULE_NAME, func_name).map_err(w)?;
        Ok(Func(func))
    }

    fn zeros(&self, len: usize) -> PyResult<Slice> {
        let slice = self.0.zeros(len).map_err(w)?;
        Ok(Slice(Arc::new(slice)))
    }

    fn slice(&self, vs: Vec<f32>) -> PyResult<Slice> {
        let slice = self.0.slice_from_values(&vs).map_err(w)?;
        Ok(Slice(Arc::new(slice)))
    }

    fn synchronize(&self) -> PyResult<()> {
        self.0.synchronize().map_err(w)
    }
}

#[pymodule]
#[pyo3(name = "ug")]
fn mod_(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Func>()?;
    m.add_class::<Device>()?;
    m.add_class::<Slice>()?;
    Ok(())
}
