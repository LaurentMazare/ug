use crate::block::{Block, Id};
use crate::lang::{self, op::Ast, ssa};
use crate::{bail, Result, Shape};
use ssa::{DType, Instr as SsaI};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimSize {
    pub dim: usize,
    /// size is the number of worker processes on the specified dim.
    pub size: usize,
}

// Default for bool is false.
#[derive(Debug, Clone, Default)]
pub struct Opts {
    local: Option<DimSize>,
    global: Option<DimSize>,
}

impl Opts {
    pub fn with_global(mut self, dim: usize, size: usize) -> Self {
        self.global = Some(DimSize { dim, size });
        self
    }

    pub fn with_local(mut self, dim: usize, size: usize) -> Self {
        self.local = Some(DimSize { dim, size });
        self
    }

    pub fn global(&self) -> &Option<DimSize> {
        &self.global
    }

    pub fn local(&self) -> &Option<DimSize> {
        &self.local
    }
}

// This represents the formula to compute the index based on some variables.
// The current structure is a pure tree so could result in exponential complexity however formulas
// are likely to be very simple so probably good to start with? If not we could update this to be a
// DAG and avoid recomputing the parts that have already been generated.
#[derive(Debug, Clone)]
enum IndexFormula {
    Id(Id),
    Const(usize),
    Add(Box<IndexFormula>, Box<IndexFormula>),
    Mul(Box<IndexFormula>, usize),
    Div(Box<IndexFormula>, usize),
    Mod(Box<IndexFormula>, usize),
}

impl From<Id> for IndexFormula {
    fn from(value: Id) -> Self {
        Self::Id(value)
    }
}

impl From<usize> for IndexFormula {
    fn from(value: usize) -> Self {
        Self::Const(value)
    }
}

impl IndexFormula {
    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }

    #[allow(unused)]
    fn mul(self, rhs: usize) -> Self {
        Self::Mul(Box::new(self), rhs)
    }

    fn div(self, rhs: usize) -> Self {
        Self::Div(Box::new(self), rhs)
    }

    fn mod_(self, rhs: usize) -> Self {
        Self::Mod(Box::new(self), rhs)
    }

    // TODO: We currently use i32 for indexes, however this prevents numerous compile time
    // optimizations and it would be better to use u32 and/or u64.
    fn eval(&self, block: &mut Block) -> Id {
        match self {
            &Self::Id(id) => id,
            &Self::Const(c) => block.push(SsaI::Const((c as i32).into())),
            Self::Add(lhs, rhs) => {
                let lhs = lhs.eval(block);
                let rhs = rhs.eval(block);
                block.binary(ssa::BinaryOp::Add, lhs, rhs, DType::I32)
            }
            Self::Mul(lhs, rhs) => {
                let lhs = lhs.eval(block);
                block.mul(lhs, *rhs as i32)
            }
            Self::Div(lhs, 1) => lhs.eval(block),
            Self::Mod(_lhs, 1) => block.push(SsaI::Const(0i32.into())),
            Self::Div(lhs, rhs) => {
                let lhs = lhs.eval(block);
                block.binary(ssa::BinaryOp::Div, lhs, *rhs as i32, DType::I32)
            }
            Self::Mod(lhs, rhs) => {
                let lhs = lhs.eval(block);
                block.binary(ssa::BinaryOp::Mod, lhs, *rhs as i32, DType::I32)
            }
        }
    }
}

/// Indexes represent the way to access the local data using the indexes from the top-level call
/// to the lower function.
/// The length of Indexes should match the shape for the local data and each IndexFormula value
/// gives the index on this dimension as offset + sum ids.1 * ids.0
#[derive(Debug, Clone)]
struct Indexes(Vec<IndexFormula>);

impl Indexes {
    fn layout_op(
        &self,
        op: &crate::lang::op::LayoutOp,
        shape: &Shape,
        arg_shape: &Shape,
    ) -> Result<Self> {
        use crate::lang::op::LayoutOp as L;
        let mut idxs = self.0.clone();
        match op {
            L::Broadcast { inserted_dims, broadcasted_dims } => {
                for dim in broadcasted_dims.iter() {
                    if *dim >= idxs.len() {
                        bail!("unexpected dim for broadcast, {dim} {:?}", shape)
                    }
                    idxs[*dim] = 0.into()
                }
                for _ in 0..*inserted_dims {
                    idxs.remove(0);
                }
            }
            &L::Narrow { dim, offset } => {
                if dim >= idxs.len() {
                    bail!("unexpected dim for narrow, {dim} {:?}", shape)
                }
                idxs[dim] = idxs[dim].clone().add(offset.into())
            }
            &L::Transpose { dim1, dim2 } => {
                if dim1 >= idxs.len() || dim2 >= idxs.len() {
                    bail!("unexpected dims for transpose {dim1} {dim2}, {:?}", shape)
                }
                idxs.swap(dim1, dim2)
            }
            &L::SplitDim { dim, lhs, rhs } => {
                if dim >= arg_shape.rank() {
                    bail!("unexpected split dim {dim} src {shape:?}")
                }
                if lhs >= shape.rank() || rhs >= shape.rank() || lhs == rhs {
                    bail!("unexpected split dims {lhs}x{rhs} dst {shape:?}")
                }
                let dims = shape.dims();
                let (l, r) = if lhs < rhs {
                    let rhs = idxs.remove(rhs);
                    let lhs = idxs.remove(lhs);
                    (lhs, rhs)
                } else {
                    let lhs = idxs.remove(lhs);
                    let rhs = idxs.remove(rhs);
                    (lhs, rhs)
                };
                idxs.insert(dim, l.mul(dims[rhs]).add(r))
            }
            &L::MergeDims { dim, lhs, rhs } => {
                if dim >= shape.rank() {
                    bail!("unexpected merge dim {dim} dst {shape:?}")
                }
                if lhs >= arg_shape.rank() || rhs >= arg_shape.rank() || lhs == rhs {
                    bail!("unexpected merge dims {lhs}x{rhs} src {arg_shape:?}")
                }
                let arg_dims = arg_shape.dims();
                let idx = idxs.remove(dim);
                // Use the C layout convention, lhs is on the left side so its stride
                // is multiplied by the size of rhs.
                let lhs_idx =
                    if arg_dims[lhs] <= 1 { 0.into() } else { idx.clone().div(arg_dims[rhs]) };
                let rhs_idx = if arg_dims[rhs] <= 1 { 0.into() } else { idx.mod_(arg_dims[rhs]) };
                if lhs < rhs {
                    idxs.insert(lhs, lhs_idx);
                    idxs.insert(rhs, rhs_idx);
                } else {
                    idxs.insert(rhs, rhs_idx);
                    idxs.insert(lhs, lhs_idx);
                }
            }
        };
        Ok(Self(idxs))
    }
}

impl lang::op::Layout {
    fn lower(&self, idxs: &Indexes) -> Result<(Id, Block)> {
        let strides = self.strides();
        if idxs.0.len() != strides.len() {
            bail!("len mismatch between strides {self:?} and idxs {idxs:?}")
        }
        let mut acc_id = None;
        let mut block = Block::empty();
        for (idx, &stride) in idxs.0.iter().zip(strides.iter()) {
            let dim_id = idx.eval(&mut block);
            let dim_id = block.mul(dim_id, stride as i32);
            let new_id = match acc_id {
                Some(acc_id) => block.binary(ssa::BinaryOp::Add, dim_id, acc_id, DType::I32),
                None => dim_id,
            };
            acc_id = Some(new_id)
        }
        match acc_id {
            Some(acc_id) => Ok((acc_id, block)),
            None => {
                let acc_id = Id::new();
                let block = Block::new(vec![(acc_id, SsaI::Const(0i32.into()))]);
                Ok((acc_id, block))
            }
        }
    }
}

impl lang::op::ReduceOp {
    fn init_value(&self, dtype: DType) -> Result<ssa::Const> {
        let value = match (self, dtype) {
            (Self::Sum, dt) => ssa::Const::zero(dt),
            (Self::Min, dt) => ssa::Const::max_value(dt),
            (Self::Max, dt) => ssa::Const::min_value(dt),
        };
        Ok(value)
    }
    fn fold_op(&self) -> lang::op::BinaryOp {
        match self {
            Self::Sum => lang::BinaryOp::Add,
            Self::Max => lang::BinaryOp::Max,
            Self::Min => lang::BinaryOp::Min,
        }
    }
}

// Simple optimization that extract the constant bits that do not depend of the index on
// some specific dim so that these can be evaluated out of loop.
fn extract_const(ast: &Ast, dim: usize) -> Result<(Vec<(Id, Ast)>, Ast)> {
    fn walk(ast: &Ast, tgt_dim: usize, accs: &mut Vec<(Id, Ast)>) -> Result<Ast> {
        use lang::op::AstInner as A;
        let ast = match ast.inner.as_ref() {
            A::Id { .. } | A::Load { .. } | A::Const(_) => ast.clone(),
            A::Reduce { op, arg, dim } => {
                if *dim == tgt_dim {
                    let src = Id::new();
                    accs.push((src, ast.clone()));
                    let inner = A::Id { src };
                    Ast {
                        inner: std::sync::Arc::new(inner),
                        dtype: ast.dtype(),
                        shape: ast.shape().clone(),
                    }
                } else {
                    let arg = walk(arg, tgt_dim, accs)?;
                    lang::op::reduce(*op, arg, *dim)?
                }
            }
            A::Unary { op, arg } => {
                let arg = walk(arg, tgt_dim, accs)?;
                lang::op::unary(*op, arg)?
            }
            A::Binary { op, lhs, rhs } => {
                let lhs = walk(lhs, tgt_dim, accs)?;
                let rhs = walk(rhs, tgt_dim, accs)?;
                lang::op::binary(*op, lhs, rhs)?
            }
            A::Layout { op, arg } => {
                let arg = walk(arg, tgt_dim, accs)?;
                let inner = A::Layout { arg, op: op.clone() };
                Ast {
                    inner: std::sync::Arc::new(inner),
                    dtype: ast.dtype(),
                    shape: ast.shape().clone(),
                }
            }
        };
        Ok(ast)
    }
    let mut accs = vec![];
    let ast = walk(ast, dim, &mut accs)?;
    Ok((accs, ast))
}

impl Ast {
    fn lower(
        &self,
        idxs: &Indexes,
        opts: &Opts,
        per_arg: &std::collections::HashMap<lang::ArgId, ssa::VarId>,
    ) -> Result<(Id, Block)> {
        use lang::op::AstInner as A;
        // TODO: Manage to enable the following, currently it doesn't work on at least consts.
        // if idxs.0.len() != self.shape().rank() {
        //     crate::bail!("internal error, idxs {idxs:?}, shape {:?}", self.shape())
        // }
        let dtype = self.dtype;
        let dst_block = match self.inner.as_ref() {
            A::Load { src, layout } => {
                let dst_i = Id::new();
                let ptr_i = match per_arg.get(src) {
                    None => bail!("unknown arg {src:?}"),
                    Some(id) => *id,
                };
                let (off_i, off_b) = layout.lower(idxs)?;
                let load = SsaI::Load { src: ptr_i, dtype, offset: off_i.to_a() };
                let mut off_b = off_b.0;
                off_b.push((dst_i, load));
                (dst_i, Block(off_b))
            }
            A::Layout { arg, op } => {
                let idxs = idxs.layout_op(op, &self.shape, &arg.shape)?;
                arg.lower(&idxs, opts, per_arg)?
            }
            A::Const(c) => {
                let dst_i = Id::new();
                (dst_i, Block::new(vec![(dst_i, SsaI::Const(*c))]))
            }
            A::Unary { op, arg } => {
                let dst_i = Id::new();
                let (arg_i, arg_b) = arg.lower(idxs, opts, per_arg)?;
                let mut arg_b = arg_b.0;
                arg_b.push((dst_i, SsaI::Unary { op: *op, arg: arg_i.to_a(), dtype }));
                (dst_i, Block(arg_b))
            }
            A::Reduce { op, arg, dim } => {
                let dst_i = Id::new();
                let mut block = Block::empty();

                let (const_bits, arg) = extract_const(arg, *dim)?;
                for (exp_id, const_bit) in const_bits.iter() {
                    let dtype = const_bit.dtype();
                    let (dst_id, const_bit) = const_bit.lower(idxs, opts, per_arg)?;
                    block.0.extend_from_slice(const_bit.0.as_slice());
                    block.0.push((
                        *exp_id,
                        SsaI::Unary { op: lang::UnaryOp::Id, arg: dst_id.to_a(), dtype },
                    ));
                }

                // TODO(laurent): generalize to other dim as long as the values in arg do not
                // depend on the dim.
                if opts.local.map_or(false, |v| v.dim == *dim) {
                    let (arg_i, arg_b) = arg.lower(idxs, opts, per_arg)?;
                    block.0.extend_from_slice(&arg_b.0);
                    block.0.push((dst_i, SsaI::ReduceLocal { op: *op, arg: arg_i.to_a(), dtype }))
                } else {
                    let init_value = op.init_value(self.dtype)?;
                    let fold_op = op.fold_op();

                    let define_acc = SsaI::DefineAcc(init_value);
                    block.0.push((dst_i, define_acc));
                    let reduce_len = match arg.shape.dims().get(*dim) {
                        None => {
                            bail!("unexpected dim for reduce, {dim} {:?}", self.shape)
                        }
                        Some(v) => *v,
                    };
                    let r = block.range(0, reduce_len as i32);

                    let mut reduce_idxs = idxs.clone();
                    reduce_idxs.0[*dim] = r.id().into();
                    let (arg_i, arg_b) = arg.lower(&reduce_idxs, opts, per_arg)?;
                    block.0.extend_from_slice(&arg_b.0);
                    let fold_op = SsaI::Binary {
                        op: fold_op,
                        lhs: dst_i.to_a(),
                        rhs: arg_i.to_a(),
                        dtype: self.dtype,
                    };
                    let src_id = block.push(fold_op);
                    block.push(SsaI::Assign { dst: dst_i.to_varid(), src: src_id.to_a() });
                    block.end_range(r)?;
                }
                (dst_i, block)
            }
            A::Binary { op, lhs, rhs } => {
                let dst_i = Id::new();
                let (lhs_i, lhs_b) = lhs.lower(idxs, opts, per_arg)?;
                let (rhs_i, rhs_b) = rhs.lower(idxs, opts, per_arg)?;
                let op = SsaI::Binary { op: *op, dtype, lhs: lhs_i.to_a(), rhs: rhs_i.to_a() };
                let instrs = [lhs_b.0.as_slice(), rhs_b.0.as_slice(), &[(dst_i, op)]].concat();
                (dst_i, Block(instrs))
            }
            A::Id { src } => (*src, Block::empty()),
        };
        Ok(dst_block)
    }
}

impl lang::op::Kernel {
    fn lower_b(&self, opts: &Opts) -> Result<Block> {
        let mut block = Block::empty();
        let mut per_arg = std::collections::HashMap::new();
        let grid_id = opts.global().map(|dim| {
            let id = block.push(SsaI::Special(ssa::Special::GridIdx));
            (dim, id)
        });
        let local_id = opts.local().map(|dim| {
            let id = block.push(SsaI::Special(ssa::Special::LocalIdx));
            (dim, id)
        });
        for (index, arg) in self.args.iter().enumerate() {
            let dtype = match arg.type_() {
                ssa::Type::Ptr(v) => v,
                ssa::Type::Value(_) => bail!("non-pointer arguments are not supported yet"),
            };
            let id = block.push(SsaI::DefineGlobal { index, dtype });
            per_arg.insert(arg.id(), id.to_varid());
        }

        for lang::op::Store { dst, layout, value } in self.ops.iter() {
            let ptr_i = match per_arg.get(dst) {
                None => bail!("unknown arg {dst:?}"),
                Some(id) => *id,
            };
            let mut ranges = Vec::with_capacity(layout.rank());
            let mut idxs = Vec::with_capacity(layout.rank());
            for (dim_idx, &len) in layout.dims().iter().enumerate() {
                let id = match (grid_id, local_id) {
                    (Some((g, grid_id)), _) if g.dim == dim_idx => grid_id,
                    (_, Some((l, local_id))) if l.dim == dim_idx => local_id,
                    (_, _) => {
                        let r = block.range(0, len as i32);
                        let id = r.id();
                        ranges.push(r);
                        id
                    }
                };
                idxs.push(id.into())
            }
            let idxs = Indexes(idxs);

            let (off_i, off_b) = layout.lower(&idxs)?;
            block.0.extend_from_slice(off_b.0.as_slice());

            let (src_i, src_b) = value.lower(&idxs, opts, &per_arg)?;
            block.0.extend_from_slice(src_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i.to_a(),
                value: src_i.to_a(),
                dtype: value.dtype,
            };
            block.push(store);
            for r in ranges.into_iter().rev() {
                block.end_range(r)?;
            }
        }
        Ok(block)
    }

    pub fn lower(&self, opts: &Opts) -> Result<ssa::Kernel> {
        let block = self.lower_b(opts)?;
        let instrs = block.relocate()?;
        let args = self.args.iter().enumerate().map(|(i, a)| (*a, i)).collect();
        Ok(ssa::Kernel::new(instrs, args))
    }
}
