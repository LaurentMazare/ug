use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    Add,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Exp,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DType {
    PtrF32,
    PtrI32,
    F32,
    I32,
}

impl DType {
    pub fn bytes(&self) -> usize {
        match self {
            Self::PtrF32 | Self::PtrI32 => 8,
            Self::F32 | Self::I32 => 4,
        }
    }
}

/// Unique identifier for arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArgId(usize);

impl ArgId {
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

/// Unique identifier for nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(usize);

impl NodeId {
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

/// Unique identifier for index nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IndexNodeId(usize);

impl IndexNodeId {
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScalarConst {
    Ptr(u64),
    I32(i32),
    F32(f32),
}

impl From<f32> for ScalarConst {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<i32> for ScalarConst {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArgType {
    Ptr,
    I32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Arg {
    id: ArgId,
    type_: ArgType,
}

impl PartialOrd for Arg {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Arg {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl Arg {
    pub fn new(type_: ArgType) -> Self {
        let id = ArgId::new();
        Self { id, type_ }
    }

    pub fn id(&self) -> ArgId {
        self.id
    }

    pub fn type_(&self) -> ArgType {
        self.type_
    }
}

#[derive(Debug, Clone)]
pub enum IndexExpr {
    ProgramId,
    Const(usize),
    Add(IndexExprNode, IndexExprNode),
    Mul(IndexExprNode, IndexExprNode),
}

#[derive(Debug, Clone)]
pub(crate) struct IndexExprNodeInner {
    #[allow(unused)]
    pub(crate) id: IndexNodeId,
    pub(crate) expr: IndexExpr,
}

#[derive(Debug, Clone)]
pub struct IndexExprNode {
    pub(crate) inner: Arc<IndexExprNodeInner>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExprNodeInner {
    #[allow(unused)]
    pub(crate) id: NodeId,
    pub(crate) expr: Expr,
}

impl IndexExprNode {
    fn from_expr(expr: IndexExpr) -> Self {
        let id = IndexNodeId::new();
        let inner = IndexExprNodeInner { id, expr };
        Self { inner: Arc::new(inner) }
    }

    pub fn cst(v: usize) -> Self {
        Self::from_expr(IndexExpr::Const(v))
    }

    pub fn program_id() -> Self {
        Self::from_expr(IndexExpr::ProgramId)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_expr(IndexExpr::Add(self.clone(), rhs.clone()))
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Self::from_expr(IndexExpr::Mul(self.clone(), rhs.clone()))
    }

    pub fn as_const(&self) -> Option<usize> {
        match &self.inner.as_ref().expr {
            IndexExpr::Const(c) => Some(*c),
            IndexExpr::ProgramId => None,
            IndexExpr::Add(lhs, rhs) => lhs.as_const().zip(rhs.as_const()).map(|(u, v)| u + v),
            IndexExpr::Mul(lhs, rhs) => lhs.as_const().zip(rhs.as_const()).map(|(u, v)| u * v),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExprNode {
    pub(crate) inner: Arc<ExprNodeInner>,
}

#[derive(Debug, Clone)]
pub struct StridedSlice {
    ptr: Arg,
    offset: IndexExprNode,
    len: IndexExprNode,
    stride: IndexExprNode,
}

impl StridedSlice {
    pub fn ptr(&self) -> &Arg {
        &self.ptr
    }
    pub fn offset(&self) -> &IndexExprNode {
        &self.offset
    }
    pub fn len(&self) -> &IndexExprNode {
        &self.len
    }
    pub fn stride(&self) -> &IndexExprNode {
        &self.stride
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    ScalarConst(ScalarConst),
    Range(usize, usize),
    Load(StridedSlice),
    Unary(UnaryOp, ExprNode),
    Binary(BinaryOp, ExprNode, ExprNode),
}

// Language in the style of triton.
#[derive(Debug, Clone)]
pub enum Ops {
    Store { dst: StridedSlice, src: ExprNode },
}

impl ExprNode {
    fn from_expr(expr: Expr) -> Self {
        let id = NodeId::new();
        let inner = ExprNodeInner { id, expr };
        Self { inner: Arc::new(inner) }
    }

    pub fn cst<C: Into<ScalarConst>>(c: C) -> Self {
        Self::from_expr(Expr::ScalarConst(c.into()))
    }

    pub fn load(
        ptr: &Arg,
        offset: &IndexExprNode,
        len: &IndexExprNode,
        stride: &IndexExprNode,
    ) -> Self {
        let ss = StridedSlice {
            ptr: *ptr,
            offset: offset.clone(),
            len: len.clone(),
            stride: stride.clone(),
        };
        Self::from_expr(Expr::Load(ss))
    }

    pub fn unary(&self, op: UnaryOp) -> Self {
        Self::from_expr(Expr::Unary(op, self.clone()))
    }

    pub fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        Self::from_expr(Expr::Binary(op, self.clone(), rhs.clone()))
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::from_expr(Expr::Binary(BinaryOp::Add, self.clone(), rhs.clone()))
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Self::from_expr(Expr::Binary(BinaryOp::Mul, self.clone(), rhs.clone()))
    }

    pub fn all_args(&self, args: &mut HashSet<Arg>) {
        match &self.inner.expr {
            Expr::Load(ss) => {
                args.insert(*ss.ptr());
            }
            Expr::Binary(_, lhs, rhs) => {
                lhs.all_args(args);
                rhs.all_args(args);
            }
            Expr::Unary(_, e) => e.all_args(args),
            Expr::Range(..) | Expr::ScalarConst(_) => {}
        }
    }
}

impl Ops {
    pub fn store(
        dst_ptr: &Arg,
        dst_offset: &IndexExprNode,
        dst_len: &IndexExprNode,
        dst_stride: &IndexExprNode,
        src: &ExprNode,
    ) -> Self {
        let dst = StridedSlice {
            ptr: *dst_ptr,
            offset: dst_offset.clone(),
            len: dst_len.clone(),
            stride: dst_stride.clone(),
        };
        Self::Store { dst, src: src.clone() }
    }

    pub fn src(&self) -> &ExprNode {
        match self {
            Self::Store { dst: _, src } => src,
        }
    }

    pub fn dst(&self) -> &StridedSlice {
        match self {
            Self::Store { dst, src: _ } => dst,
        }
    }

    pub fn all_args(&self, args: &mut HashSet<Arg>) {
        args.insert(*self.dst().ptr());
        self.src().all_args(args);
    }
}

#[derive(Debug, Clone)]
pub struct Kernel {
    #[allow(unused)]
    pub(crate) name: String,
    pub(crate) args: Vec<Arg>,
    pub(crate) ops: Vec<Ops>,
}

impl Kernel {
    pub fn new(name: String, args: Vec<Arg>, ops: Vec<Ops>) -> Self {
        Self { name, args, ops }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlopsMem {
    pub flops: usize,
    pub mem_in_bytes: usize,
}

// AST version of the SSA ops
pub mod op {
    pub use super::{Arg, ArgId, ArgType, BinaryOp, DType, ReduceOp, UnaryOp};
    use anyhow::Result;
    use std::sync::Arc;

    // TODO(laurent): Split the layout type in a separate module.
    // TODO(laurent): Dedicated type for shape, similar to candle.
    #[derive(Debug, Clone)]
    pub struct Layout {
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    }

    impl Layout {
        pub fn from_shape(shape: &[usize]) -> Self {
            let mut strides = vec![];
            let mut stride = 1;
            for l in shape.iter().rev() {
                strides.push(stride);
                stride *= l
            }
            strides.reverse();
            let shape = shape.to_vec();
            Self { shape, strides, offset: 0 }
        }

        pub fn num_elements(&self) -> usize {
            self.shape.iter().product()
        }

        pub fn dims(&self) -> usize {
            self.shape.len()
        }

        pub fn strides(&self) -> &[usize] {
            self.strides.as_slice()
        }

        pub fn shape(&self) -> &[usize] {
            self.shape.as_slice()
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn c_contiguous(&self) -> bool {
            let mut prod_l = 1;
            for (&s, &l) in self.strides.iter().zip(self.shape.iter()).rev() {
                if s != prod_l {
                    return false;
                }
                prod_l *= l
            }
            true
        }
    }

    #[derive(Debug, Clone)]
    pub struct Ast {
        pub(crate) inner: Arc<AstInner>,
        pub(crate) dtype: DType,
        pub(crate) shape: Vec<usize>,
    }

    #[derive(Debug, Clone)]
    pub enum AstInner {
        Load { src: ArgId, layout: Layout },
        Reduce { op: ReduceOp, arg: Ast, axis: usize },
        Unary { op: UnaryOp, arg: Ast },
        Binary { op: BinaryOp, lhs: Ast, rhs: Ast },
        // TODO(laurent): Add some reshape/transpose/const...
    }

    pub fn load(src: ArgId, layout: Layout, dtype: DType) -> Result<Ast> {
        let shape = layout.shape.clone();
        let inner = AstInner::Load { src, layout };
        Ok(Ast { inner: Arc::new(inner), dtype, shape })
    }

    pub fn unary(op: UnaryOp, arg: Ast) -> Result<Ast> {
        let dtype = arg.dtype;
        let shape = arg.shape.clone();
        let inner = AstInner::Unary { op, arg };
        Ok(Ast { inner: Arc::new(inner), dtype, shape })
    }

    pub fn reduce(op: ReduceOp, arg: Ast, axis: usize) -> Result<Ast> {
        let dtype = arg.dtype;
        let mut shape = arg.shape.clone();
        if axis >= shape.len() {
            anyhow::bail!("no axis {axis} in shape {shape:?}")
        }
        shape.remove(axis);
        let inner = AstInner::Reduce { op, arg, axis };
        Ok(Ast { inner: Arc::new(inner), dtype, shape })
    }

    pub fn binary(op: BinaryOp, lhs: Ast, rhs: Ast) -> Result<Ast> {
        let dtype = if lhs.dtype != rhs.dtype {
            anyhow::bail!("dtype mismatch in {op:?}, lhs: {:?}, rhs: {:?}", lhs.dtype, rhs.dtype)
        } else {
            lhs.dtype
        };
        // TODO(laurent): check the shape, should broadcast be implicit or not?
        if lhs.shape != rhs.shape {
            anyhow::bail!("shape mismatch in {op:?}, lhs: {:?}, rhs: {:?}", lhs.shape, rhs.shape)
        }
        let shape = lhs.shape.clone();
        let inner = AstInner::Binary { op, lhs, rhs };
        Ok(Ast { inner: Arc::new(inner), dtype, shape })
    }

    impl Ast {
        pub fn dtype(&self) -> DType {
            self.dtype
        }

        pub fn shape(&self) -> &[usize] {
            self.shape.as_slice()
        }
    }

    #[derive(Debug, Clone)]
    pub struct Store {
        pub(crate) dst: ArgId,
        pub(crate) layout: Layout,
        pub(crate) value: Ast,
    }

    pub fn store(dst: ArgId, layout: Layout, value: Ast) -> Result<Store> {
        Ok(Store { dst, layout, value })
    }

    impl Store {
        pub fn dtype(&self) -> DType {
            self.value.dtype
        }
    }

    #[derive(Debug, Clone)]
    pub struct Kernel {
        #[allow(unused)]
        pub(crate) name: String,
        pub(crate) args: Vec<Arg>,
        pub(crate) ops: Vec<Store>,
    }

    impl Kernel {
        pub fn new(name: String, args: Vec<Arg>, ops: Vec<Store>) -> Self {
            Self { name, args, ops }
        }
    }
}

// Very untyped almost SSA language.
// There are no phi symbols, instead Range is used.
// This is currently close to the UOps setup from tinygrad:
// https://github.com/tinygrad/tinygrad/blob/13846930cd43b1cfd8f7bb2967529f08c08cb6d6/tinygrad/ops.py#L98
pub mod ssa {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    pub use super::{BinaryOp, DType, UnaryOp};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub struct VarId(usize);

    impl VarId {
        pub fn new(v: usize) -> Self {
            Self(v)
        }

        pub fn as_usize(&self) -> usize {
            self.0
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub enum Const {
        I32(i32),
        F32(f32),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub enum Special {
        LocalIdx,
        GridIdx,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Instr {
        DefineAcc(Const),
        DefineGlobal { index: usize, dtype: DType },
        DefineLocal { size: usize, dtype: DType },
        Special(Special),
        Const(Const),
        Unary { op: UnaryOp, arg: VarId, dtype: DType },
        Binary { op: BinaryOp, lhs: VarId, rhs: VarId, dtype: DType },
        // TODO(laurent): using an absolute end_idx is convenient for jumping
        // out of the loop but is problematic when combining blocks. Maybe we
        // should switch to a relative one?
        Range { lo: VarId, up: VarId, end_idx: usize },
        Load { src: VarId, offset: VarId, dtype: DType },
        Assign { dst: VarId, src: VarId },
        EndRange { start_idx: usize },
        Store { dst: VarId, offset: VarId, value: VarId, dtype: DType },
        Barrier,
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct Kernel {
        pub instrs: Vec<Instr>,
    }

    impl Kernel {
        pub fn flops_mem_per_thread(&self) -> Result<super::FlopsMem> {
            let mut flops = 0usize;
            let mut mem = 0usize;
            let mut mults = vec![];
            let mut mult = 1usize;
            for instr in self.instrs.iter() {
                match instr {
                    Instr::Load { src: _, offset: _, dtype }
                    | Instr::Store { dst: _, offset: _, value: _, dtype } => {
                        mem += mult * dtype.bytes()
                    }
                    Instr::Range { lo, up, end_idx: _ } => {
                        mults.push(mult);
                        let lo = match self.instrs[lo.0] {
                            Instr::Const(Const::I32(lo)) => lo,
                            _ => anyhow::bail!("range lo is not a const"),
                        };
                        let up = match self.instrs[up.0] {
                            Instr::Const(Const::I32(up)) => up,
                            _ => anyhow::bail!("range up is not a const"),
                        };
                        mult *= (up - lo).max(0) as usize;
                    }
                    Instr::EndRange { .. } => match mults.pop() {
                        None => anyhow::bail!("unexpected EndRange"),
                        Some(m) => mult = m,
                    },
                    Instr::Unary { .. } | Instr::Binary { .. } => flops += mult,
                    Instr::DefineGlobal { .. }
                    | Instr::DefineLocal { .. }
                    | Instr::DefineAcc(_)
                    | Instr::Special(_)
                    | Instr::Assign { .. }
                    | Instr::Const(_)
                    | Instr::Barrier => {}
                }
            }
            Ok(super::FlopsMem { flops, mem_in_bytes: mem })
        }
    }

    impl std::fmt::Debug for Kernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for (var_id, instr) in self.instrs.iter().enumerate() {
                writeln!(f, "{var_id:03} {instr:?}")?
            }
            Ok(())
        }
    }
}
