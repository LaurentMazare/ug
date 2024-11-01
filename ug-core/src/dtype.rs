use crate::{bail, Const, Result};
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

// Poor man's GADT...
#[derive(Debug, Copy, Clone)]
pub enum CpuStorageRef<'a> {
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

#[derive(Debug)]
pub enum CpuStorageRefMut<'a> {
    BF16(&'a mut [bf16]),
    F16(&'a mut [f16]),
    F32(&'a mut [f32]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I32,
    I64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError(String);

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot parse '{}' as a dtype", self.0)
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}

impl DType {
    /// String representation for dtypes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    /// The size used by each element in bytes, i.e. 4 for `F32`.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::I32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::I32 | Self::I64 => true,
            Self::BF16 | Self::F16 | Self::F32 => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::I32 | Self::I64 => false,
            Self::BF16 | Self::F16 | Self::F32 => true,
        }
    }
}

pub trait WithDType: Copy + Clone + 'static + num::Zero + num::One {
    const DTYPE: DType;

    fn to_const(self) -> Const;
    fn from_const(data: Const) -> Result<Self>;
    fn to_cpu_storage(data: &[Self]) -> CpuStorageRef<'_>;
    fn from_cpu_storage(data: CpuStorageRef<'_>) -> Result<&[Self]>;
    fn to_cpu_storage_mut(data: &mut [Self]) -> CpuStorageRefMut<'_>;
    fn from_cpu_storage_mut(data: CpuStorageRefMut<'_>) -> Result<&mut [Self]>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn to_const(self) -> Const {
                Const::$dtype(self)
            }

            fn from_const(data: Const) -> Result<Self> {
                match data {
                    Const::$dtype(data) => Ok(data),
                    _ => {
                        bail!(
                            "unexpected dtype for const, expected {:?}, got {:?}",
                            Self::DTYPE,
                            data.dtype()
                        )
                    }
                }
            }

            fn to_cpu_storage_mut(data: &mut [Self]) -> CpuStorageRefMut<'_> {
                CpuStorageRefMut::$dtype(data)
            }

            fn from_cpu_storage_mut(data: CpuStorageRefMut<'_>) -> Result<&mut [Self]> {
                match data {
                    CpuStorageRefMut::$dtype(data) => Ok(data),
                    _ => {
                        bail!(
                            "unexpected dtype, expected {:?}, got {:?}",
                            Self::DTYPE,
                            data.dtype()
                        )
                    }
                }
            }
            fn to_cpu_storage(data: &[Self]) -> CpuStorageRef<'_> {
                CpuStorageRef::$dtype(data)
            }

            fn from_cpu_storage(data: CpuStorageRef<'_>) -> Result<&[Self]> {
                match data {
                    CpuStorageRef::$dtype(data) => Ok(data),
                    _ => {
                        bail!(
                            "unexpected dtype, expected {:?}, got {:?}",
                            Self::DTYPE,
                            data.dtype()
                        )
                    }
                }
            }
        }
    };
}
with_dtype!(bf16, BF16);
with_dtype!(f16, F16);
with_dtype!(f32, F32);
with_dtype!(i32, I32);
with_dtype!(i64, I64);

impl CpuStorage {
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            Self::BF16(s) => s.as_mut_ptr() as *mut std::ffi::c_void,
            Self::F16(s) => s.as_mut_ptr() as *mut std::ffi::c_void,
            Self::F32(s) => s.as_mut_ptr() as *mut std::ffi::c_void,
            Self::I32(s) => s.as_mut_ptr() as *mut std::ffi::c_void,
            Self::I64(s) => s.as_mut_ptr() as *mut std::ffi::c_void,
        }
    }

    pub fn as_ptr(&mut self) -> *const std::ffi::c_void {
        match self {
            Self::BF16(s) => s.as_ptr() as *const std::ffi::c_void,
            Self::F16(s) => s.as_ptr() as *const std::ffi::c_void,
            Self::F32(s) => s.as_ptr() as *const std::ffi::c_void,
            Self::I32(s) => s.as_ptr() as *const std::ffi::c_void,
            Self::I64(s) => s.as_ptr() as *const std::ffi::c_void,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::BF16(s) => s.len(),
            Self::F16(s) => s.len(),
            Self::F32(s) => s.len(),
            Self::I32(s) => s.len(),
            Self::I64(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
        }
    }

    pub fn as_ref(&self) -> CpuStorageRef<'_> {
        match self {
            Self::BF16(v) => CpuStorageRef::BF16(v.as_slice()),
            Self::F16(v) => CpuStorageRef::F16(v.as_slice()),
            Self::F32(v) => CpuStorageRef::F32(v.as_slice()),
            Self::I32(v) => CpuStorageRef::I32(v.as_slice()),
            Self::I64(v) => CpuStorageRef::I64(v.as_slice()),
        }
    }
}

impl CpuStorageRef<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
        }
    }
}

impl CpuStorageRefMut<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
        }
    }
}
