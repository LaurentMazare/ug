use crate::{Const, DType, Device, Layout, Result, Shape};

/// Unique identifier for LazyBuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(usize);

impl Id {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub enum LayoutOp {
    Reshape,
    Broadcast,
    Noop,
}

pub enum Op<D: Device> {
    Unary(crate::lang::UnaryOp, LazyBuffer<D>),
    Binary(crate::lang::BinaryOp, LazyBuffer<D>, LazyBuffer<D>),
    Reduce(crate::lang::ReduceOp, LazyBuffer<D>, usize),
    Const(crate::lang::Const),
    // TODO: maybe the following should be an Arc<Mutex<...>> or similar so that it can easily be
    // modified?
    Copy(crate::CpuStorage),
    Layout(LayoutOp, LazyBuffer<D>),
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
    id: Id,
    // A Mutex here has some runtime overhead when lots of buffers are
    // used but allows the structure to be shared between threads.
    data: std::sync::Mutex<Option<D::Slice>>,
    op: Op<D>,
    dtype: crate::DType,
    layout: Layout,
    device: D,
}

impl<D: Device> LazyBuffer<D> {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn maybe_allocate_uninit(&self) -> Result<()> {
        let mut data = self.data.lock()?;
        if data.is_none() {
            // TODO: This should only apply to C contiguous tensors.
            let nels = self.layout.num_elements();
            let v = match self.dtype {
                DType::BF16 => self.device.allocate_uninit::<half::bf16>(nels)?,
                DType::F16 => self.device.allocate_uninit::<half::f16>(nels)?,
                DType::F32 => self.device.allocate_uninit::<f32>(nels)?,
                DType::I32 => self.device.allocate_uninit::<i32>(nels)?,
                DType::I64 => self.device.allocate_uninit::<i64>(nels)?,
            };
            *data = Some(v)
        }
        Ok(())
    }

    pub fn op(&self) -> &Op<D> {
        &self.op
    }

    pub fn realized(&self) -> bool {
        let data = self.data.lock().unwrap();
        data.is_some()
    }

    pub fn data(&self) -> &std::sync::Mutex<Option<D::Slice>> {
        &self.data
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
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Unary(op, self.clone()),
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
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Binary(op, self.clone(), rhs),
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
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Reduce(op, self.clone(), axis),
            dtype: self.dtype,
            device: self.device.clone(),
            layout: Layout::from_shape(shape),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn id(&self) -> Id {
        self.id
    }

    pub fn reshape<S: Into<Shape>>(&self, s: S) -> Result<Self> {
        let s: Shape = s.into();
        let dst_nel = s.num_elements();
        let src_nel = self.shape().num_elements();
        if dst_nel != src_nel {
            crate::bail!(
                "cannot reshape between {:?} ({src_nel} elts) and {s:?} ({dst_nel} elts)",
                self.shape()
            )
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Layout(LayoutOp::Reshape, self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            layout: Layout::from_shape(s),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn broadcast<S: Into<Shape>>(&self, s: S) -> Result<Self> {
        let s: Shape = s.into();
        if self.shape().rank() > s.rank() {
            crate::bail!(
                "target shape {s:?} has less dimensions than original shape {:?}",
                self.shape()
            )
        }
        let inserted_dims = s.rank() - self.shape().rank();
        for (dim_idx, &dim_len) in self.shape().dims().iter().enumerate() {
            let dim_idx = dim_idx + inserted_dims;
            if s.dims()[dim_idx] != dim_len && dim_len != 1 {
                crate::bail!("cannot broadcast from {:?} to {s:?}", self.shape())
            }
        }

        let inner = LazyBufferInner {
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Layout(LayoutOp::Broadcast, self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            layout: Layout::from_shape(s),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn cst<C: Into<Const>, S: Into<Shape>>(c: C, s: S, device: &D) -> Result<Self> {
        let c: Const = c.into();
        let s: Shape = s.into();
        let inner = LazyBufferInner {
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Const(c),
            dtype: c.dtype(),
            device: device.clone(),
            layout: Layout::from_shape(s),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }

    pub fn copy<S: Into<Shape>>(data: crate::CpuStorage, s: S, device: &D) -> Result<Self> {
        let s: Shape = s.into();
        let dtype = data.dtype();
        let inner = LazyBufferInner {
            id: Id::new(),
            data: std::sync::Mutex::new(None),
            op: Op::Copy(data),
            dtype,
            device: device.clone(),
            layout: Layout::from_shape(s),
        };
        let lb = LazyBuffer(std::sync::Arc::new(inner));
        Ok(lb)
    }
}
