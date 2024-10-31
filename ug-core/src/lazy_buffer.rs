#![allow(unused)]
use crate::dtype::WithDType;
use crate::{DType, Layout, Result, Shape};

pub trait Slice {
    type Device: Device<Slice = Self>;

    fn device(&self) -> &Self::Device;
    fn dtype(&self) -> DType;
    fn len(&self) -> usize;
    fn copy_host_to_device<DT: WithDType>(&mut self, src: &[DT]) -> Result<()>;
    fn copy_device_to_host<DT: WithDType>(&self, dst: &mut [DT]) -> Result<()>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn to_vec<DT: WithDType>(&self) -> Result<Vec<DT>> {
        let mut host = vec![DT::zero(); self.len()];
        self.copy_device_to_host(&mut host)?;
        Ok(host)
    }
}

pub trait Device: Clone {
    type Slice: Slice<Device = Self>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn allocate_uninit<DT: WithDType>(&self, len: usize) -> Result<Self::Slice>;
    fn synchronize(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub enum LayoutUpdate {
    Reshape,
    Broadcast,
}

pub enum Op<D: Device> {
    Unary(crate::lang::UnaryOp, LazyBuffer<D>),
    Binary(crate::lang::BinaryOp, LazyBuffer<D>, LazyBuffer<D>),
    Reduce(crate::lang::ReduceOp, LazyBuffer<D>, usize),
    Const(crate::lang::Const),
    // TODO: maybe the following should be an Arc<Mutex<...>> or similar so that it can easily be
    // modified.
    Copy(crate::dtype::CpuStorage),
    Layout(LayoutUpdate),
}

pub struct LazyBuffer<D: Device>(std::sync::Arc<LazyBufferInner<D>>);

impl<D: Device> Clone for LazyBuffer<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

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
    layout: Layout,
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

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    pub fn unary(&self, op: crate::lang::UnaryOp) -> Result<Self> {
        // TODO: dtype/op checks.
        let inner = LazyBufferInner {
            data: None,
            op: Some(Op::Unary(op, self.clone())),
            dtype: self.dtype,
            layout: Layout::from_shape(self.shape()),
            device: self.device.clone(),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn binary(&self, op: crate::lang::BinaryOp, rhs: Self) -> Result<Self> {
        // TODO: dtype/op/shape checks.
        let inner = LazyBufferInner {
            data: None,
            op: Some(Op::Binary(op, self.clone(), rhs)),
            dtype: self.dtype,
            device: self.device.clone(),
            layout: Layout::from_shape(self.shape()),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn reduce(&self, op: crate::lang::ReduceOp, axis: usize) -> Result<Self> {
        // TODO: dtype/op checks.
        let shape = self.shape(); // TODO: squeeze or remove axis.
        let inner = LazyBufferInner {
            data: None,
            op: Some(Op::Reduce(op, self.clone(), axis)),
            dtype: self.dtype,
            device: self.device.clone(),
            layout: Layout::from_shape(shape),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }
}
