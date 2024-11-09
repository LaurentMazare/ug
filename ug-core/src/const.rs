use crate::DType;
use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Const {
    I32(i32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
}

impl From<bf16> for Const {
    fn from(value: bf16) -> Self {
        Self::BF16(value)
    }
}

impl From<f16> for Const {
    fn from(value: f16) -> Self {
        Self::F16(value)
    }
}

impl From<i64> for Const {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<i32> for Const {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

impl From<f32> for Const {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl Const {
    pub fn dtype(&self) -> DType {
        match self {
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
        }
    }

    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16(f16::ZERO),
            DType::BF16 => Self::BF16(bf16::ZERO),
            DType::F32 => Self::F32(0f32),
            DType::I32 => Self::I32(0i32),
            DType::I64 => Self::I64(0i64),
        }
    }

    pub fn min_value(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16(f16::NEG_INFINITY),
            DType::BF16 => Self::BF16(bf16::NEG_INFINITY),
            DType::F32 => Self::F32(f32::NEG_INFINITY),
            DType::I32 => Self::I32(i32::MIN),
            DType::I64 => Self::I64(i64::MIN),
        }
    }

    pub fn max_value(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16(f16::INFINITY),
            DType::BF16 => Self::BF16(bf16::INFINITY),
            DType::F32 => Self::F32(f32::INFINITY),
            DType::I32 => Self::I32(i32::MAX),
            DType::I64 => Self::I64(i64::MAX),
        }
    }
}
