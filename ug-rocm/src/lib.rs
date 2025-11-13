// Created by: TEAM-510 | ROCm backend for ug kernel compiler
pub use rocm_rs;
pub mod code_gen;
pub mod runtime;

pub use runtime::{Device, DeviceSlice};
