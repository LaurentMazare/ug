use crate::{bail, Const, DType, Device, Dim, Result, Shape};
use std::sync::{Arc, Mutex};

type Callback<S> = Arc<dyn Fn(Vec<&mut S>) -> Result<()>>;
pub struct CustomF<S>(Callback<S>);

impl<S> std::fmt::Debug for CustomF<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<func>")
    }
}

impl<S> std::ops::Deref for CustomF<S> {
    type Target = Arc<dyn Fn(Vec<&mut S>) -> Result<()>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S> Clone for CustomF<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<S> CustomF<S> {
    pub fn new<F: 'static + Fn(Vec<&mut S>) -> Result<()>>(f: F) -> Self {
        Self(Arc::new(f))
    }
}

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

#[derive(Debug)]
pub enum Op<D: Device> {
    Unary(crate::lang::UnaryOp, LazyBuffer<D>),
    Binary(crate::lang::BinaryOp, LazyBuffer<D>, LazyBuffer<D>),
    MatMul(LazyBuffer<D>, LazyBuffer<D>, (usize, usize, usize, usize), bool),
    Reduce(crate::lang::ReduceOp, LazyBuffer<D>, usize),
    Const(crate::lang::Const),
    Copy,
    Layout(crate::lang::op::LayoutOp, LazyBuffer<D>),
    Reshape(LazyBuffer<D>),
    Custom { f: CustomF<D::Slice>, args: Vec<LazyBuffer<D>> },
    CustomIp { f: CustomF<D::Slice>, args: Vec<LazyBuffer<D>>, src: LazyBuffer<D> },
    Ssa { ssa: crate::lang::ssa::Kernel, args: Vec<LazyBuffer<D>> },
}

pub struct LazyBuffer<D: Device>(Arc<LazyBufferInner<D>>);

impl<D: Device> std::fmt::Debug for LazyBuffer<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?} {:?} {:?}]", self.id(), self.shape(), self.dtype())?;
        Ok(())
    }
}

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
    data: Arc<Mutex<Option<D::Slice>>>,
    op: Op<D>,
    dtype: crate::DType,
    /// The shape for the buffer, the buffer always uses a C style memory layout.
    shape: Shape,
    device: D,
}

impl<D: Device> LazyBuffer<D> {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn maybe_allocate_uninit(&self) -> Result<()> {
        let mut data = self.data.lock()?;
        if data.is_none() {
            let nels = self.shape.num_elements();
            let v = self.device.allocate_uninit(self.dtype, nels)?;
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

    pub fn realize(&self) -> Result<()> {
        if self.realized() {
            return Ok(());
        }
        let schedule = crate::Schedule::create_one(self)?;
        let schedule = schedule.compile()?;
        schedule.run()?;
        Ok(())
    }

    pub fn data(&self) -> &Mutex<Option<D::Slice>> {
        self.data.as_ref()
    }

    pub fn data_vec<DT: crate::WithDType>(&self) -> Result<Option<Vec<DT>>> {
        use crate::Slice;

        let data = self.data.as_ref().lock()?;
        let data = match data.as_ref() {
            None => None,
            Some(data) => {
                let mut vs = vec![DT::zero(); self.shape.num_elements()];
                D::Slice::copy_device_to_host(data, &mut vs)?;
                Some(vs)
            }
        };
        Ok(data)
    }

    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn dims0(&self) -> Result<()> {
        self.shape().dims0()
    }

    pub fn dims1(&self) -> Result<usize> {
        self.shape().dims1()
    }

    pub fn dims2(&self) -> Result<(usize, usize)> {
        self.shape().dims2()
    }

    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        self.shape().dims3()
    }

    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        self.shape().dims4()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn unary(&self, op: crate::lang::UnaryOp) -> Result<Self> {
        // TODO: dtype/op checks.
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Unary(op, self.clone()),
            dtype: self.dtype,
            shape: self.shape.clone(),
            device: self.device.clone(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn cast(&self, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Unary(crate::lang::UnaryOp::Cast, self.clone()),
            dtype,
            shape: self.shape.clone(),
            device: self.device.clone(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn binary(&self, op: crate::lang::BinaryOp, rhs: Self) -> Result<Self> {
        if self.shape != rhs.shape {
            bail!("shape mismatch in {op:?}, {:?} vs {:?}", self.shape, rhs.shape)
        }
        if self.dtype != rhs.dtype {
            bail!("dtype mismatch in {op:?}, {:?} vs {:?}", self.dtype, rhs.dtype)
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Binary(op, self.clone(), rhs),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: self.shape.clone(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    // TODO: Should this be marked as unsafe?
    pub fn alloc_uninit<S: Into<Shape>>(dtype: DType, s: S, device: &D) -> Result<Self> {
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Copy,
            dtype,
            device: device.clone(),
            shape: s.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn custom<F: 'static + Fn(Vec<&mut D::Slice>) -> Result<()>, S: Into<Shape>>(
        f: F,
        args: Vec<Self>,
        s: S,
        dtype: DType,
        device: &D,
    ) -> Result<Self> {
        let f = CustomF::new(f);
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Custom { f, args },
            dtype,
            device: device.clone(),
            shape: s.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn custom_ip<F: 'static + Fn(Vec<&mut D::Slice>) -> Result<()>>(
        &self,
        f: F,
        args: Vec<Self>,
    ) -> Result<Self> {
        let f = CustomF::new(f);
        let inner = LazyBufferInner {
            id: Id::new(),
            data: self.data.clone(),
            op: Op::CustomIp { f, args, src: self.clone() },
            dtype: self.dtype,
            device: self.device.clone(),
            shape: self.shape.clone(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn ssa<S: Into<Shape>>(
        ssa: crate::lang::ssa::Kernel,
        args: Vec<Self>,
        s: S,
        dtype: DType,
        device: &D,
    ) -> Result<Self> {
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Ssa { ssa, args },
            dtype,
            device: device.clone(),
            shape: s.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn matmul(&self, rhs: Self) -> Result<Self> {
        self.matmul_(rhs, false)
    }

    pub fn matmul_t(&self, rhs: Self) -> Result<Self> {
        self.matmul_(rhs, true)
    }

    pub fn matmul_(&self, rhs: Self, transpose: bool) -> Result<Self> {
        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        let dim = lhs_dims.len();
        let rdim = rhs_dims.len();

        if dim < 2 || rdim < 2 {
            bail!("shape mismatch in matmul {lhs_dims:?} {rhs_dims:?}")
        }

        let m = lhs_dims[dim - 2];
        let k = lhs_dims[dim - 1];
        let (k2, n) = if transpose {
            (rhs_dims[rdim - 1], rhs_dims[rdim - 2])
        } else {
            (rhs_dims[rdim - 2], rhs_dims[rdim - 1])
        };

        let lhs_bsz: usize = lhs_dims[..dim - 2].iter().product();
        let rhs_bsz: usize = rhs_dims[..rdim - 2].iter().product();
        if k != k2 || lhs_bsz != rhs_bsz {
            bail!("shape mismatch in matmul {lhs_dims:?} {rhs_dims:?}")
        }
        let bmnk = (lhs_bsz, m, n, k);
        let mut shape = lhs_dims[..dim - 2].to_vec();
        shape.push(m);
        shape.push(n);
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::MatMul(self.clone(), rhs, bmnk, transpose),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: shape.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn reduce<I: Dim>(&self, op: crate::lang::ReduceOp, dim: I) -> Result<Self> {
        // TODO: dtype/op checks.
        let shape = self.shape(); // TODO: squeeze or remove axis.
        let dim = dim.to_index(shape, "reduce")?;
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Reduce(op, self.clone(), dim),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: shape.clone(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn id(&self) -> Id {
        self.id
    }

    pub fn split_dim<I: Dim>(&self, dim: I, size1: usize, size2: usize) -> Result<Self> {
        use crate::lang::op::LayoutOp::SplitDim;
        let dim = dim.to_index(self.shape(), "split_dim")?;
        let mut dims = self.dims().to_vec();
        let size = dims.remove(dim);
        if size1 * size2 != size {
            bail!("unexpected target sizes for split_dim {dim}, {size1}x{size2} != {size}",)
        }
        dims.insert(dim, size2);
        dims.insert(dim, size1);
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Layout(SplitDim { dim, lhs: dim, rhs: dim + 1 }, self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: dims.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    /// Merge the dims dim and dim + 1 together.
    pub fn merge_dims<I: Dim>(&self, dim: I) -> Result<Self> {
        use crate::lang::op::LayoutOp::MergeDims;
        let dim = dim.to_index(self.shape(), "split_dim")?;
        if dim + 1 >= self.rank() {
            bail!("unexpected dim for merge_dims {dim} {:?}", self.shape())
        }
        let mut dims = self.dims().to_vec();
        let size_p = dims.remove(dim + 1);
        dims[dim] *= size_p;
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Layout(MergeDims { dim, lhs: dim, rhs: dim + 1 }, self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: dims.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn reshape<S: Into<Shape>>(&self, s: S) -> Result<Self> {
        let s: Shape = s.into();
        let dst_nel = s.num_elements();
        let src_nel = self.shape().num_elements();
        if dst_nel != src_nel {
            bail!(
                "cannot reshape between {:?} ({src_nel} elts) and {s:?} ({dst_nel} elts)",
                self.shape()
            )
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Reshape(self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: s,
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn broadcast<S: Into<Shape>>(&self, s: S) -> Result<Self> {
        let s: Shape = s.into();
        if self.shape().rank() > s.rank() {
            bail!("target shape {s:?} has less dimensions than original shape {:?}", self.shape())
        }
        let inserted_dims = s.rank() - self.shape().rank();
        let mut broadcasted_dims = (0..inserted_dims).collect::<Vec<_>>();
        for (dim_idx, &dim_len) in self.shape.dims().iter().enumerate() {
            let dim_idx = dim_idx + inserted_dims;
            if s.dims()[dim_idx] != dim_len {
                if dim_len == 1 {
                    broadcasted_dims.push(dim_idx)
                } else {
                    bail!("cannot broadcast from {:?} to {s:?}", self.shape)
                }
            }
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Layout(
                crate::lang::op::LayoutOp::Broadcast { inserted_dims, broadcasted_dims },
                self.clone(),
            ),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: s,
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(self.shape(), "transpose")?;
        let dim2 = dim2.to_index(self.shape(), "transpose")?;
        if dim1 == dim2 {
            return Ok(self.clone());
        }
        let mut dims = self.dims().to_vec();
        dims.swap(dim1, dim2);
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Layout(crate::lang::op::LayoutOp::Transpose { dim1, dim2 }, self.clone()),
            dtype: self.dtype,
            device: self.device.clone(),
            shape: dims.into(),
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn cst<C: TryInto<Const> + std::fmt::Debug + Copy, S: Into<Shape>>(
        c: C,
        s: S,
        device: &D,
    ) -> Result<Self> {
        let c: Const = match c.try_into() {
            Err(_) => bail!("unable to create const for {c:?}"),
            Ok(v) => v,
        };
        let s: Shape = s.into();
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(None)),
            op: Op::Const(c),
            dtype: c.dtype(),
            device: device.clone(),
            shape: s,
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn from_slice<S: Into<Shape>>(data: D::Slice, s: S) -> Result<Self> {
        use crate::Slice;

        let s: Shape = s.into();
        let device = data.device().clone();
        let dtype = data.dtype();
        if s.num_elements() != data.len() {
            bail!("unexpected number of elements {} for shape {s:?}", data.len())
        }
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(Some(data))),
            // We don't keep a hold on the Copy data here so as to reduce memory usage.
            op: Op::Copy,
            dtype,
            device: device.clone(),
            shape: s,
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }

    pub fn copy<'a, R: Into<crate::CpuStorageRef<'a>>, S: Into<Shape>>(
        data: R,
        s: S,
        device: &D,
    ) -> Result<Self> {
        use crate::CpuStorageRef as C;
        use crate::Slice;

        let data: crate::CpuStorageRef<'a> = data.into();
        let s: Shape = s.into();
        if s.num_elements() != data.len() {
            bail!("unexpected number of elements {} for shape {s:?}", data.len())
        }
        let dtype = data.dtype();
        let mut slice = unsafe { device.allocate_uninit(dtype, data.len()) }?;
        match data {
            C::BF16(data) => slice.copy_host_to_device(data)?,
            C::F16(data) => slice.copy_host_to_device(data)?,
            C::F32(data) => slice.copy_host_to_device(data)?,
            C::I32(data) => slice.copy_host_to_device(data)?,
            C::I64(data) => slice.copy_host_to_device(data)?,
        };
        let inner = LazyBufferInner {
            id: Id::new(),
            data: Arc::new(Mutex::new(Some(slice))),
            // We don't keep a hold on the Copy data here so as to reduce memory usage.
            op: Op::Copy,
            dtype,
            device: device.clone(),
            shape: s,
        };
        let lb = LazyBuffer(Arc::new(inner));
        Ok(lb)
    }
}
