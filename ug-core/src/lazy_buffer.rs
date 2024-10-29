#![allow(unused)]
use crate::dtype::WithDType;
use crate::{DType, Result};

pub trait Device {
    type Slice: Sized;

    fn allocate_uninit<DT: WithDType>(&self, len: usize) -> Result<Self::Slice>;
    fn copy_host_to_device<DT: WithDType>(src: &[DT], dst: &mut Self::Slice) -> Result<()>;
    fn copy_device_to_host<DT: WithDType>(src: &Self::Slice, dst: &mut [DT]) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}

pub enum Op<D: Device> {
    Unary(crate::lang::UnaryOp, LazyBuffer<D>),
    Binary(crate::lang::BinaryOp, LazyBuffer<D>, LazyBuffer<D>),
    Reduce(crate::lang::ReduceOp, LazyBuffer<D>, usize),
}

pub struct LazyBuffer<D: Device>(std::sync::Arc<LazyBufferInner<D>>);

impl<D: Device> std::ops::Deref for LazyBuffer<D> {
    type Target = LazyBufferInner<D>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

pub struct LazyBufferInner<D: Device> {
    data: Option<D::Slice>,
    op: Option<Op<D>>,
    dtype: crate::DType,
    device: D,
}

impl<D: Device> LazyBuffer<D> {
    pub fn realized(&self) -> bool {
        self.data.is_some()
    }

    pub fn data(&self) -> Option<&D::Slice> {
        self.data.as_ref()
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
