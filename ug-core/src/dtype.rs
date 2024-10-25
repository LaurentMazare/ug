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
