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

#[derive(Debug, Clone)]
struct IndexFormula {
    ids: Vec<(Id, usize)>,
    offset: usize,
}

/// Indexes represent the way to access the local data using the indexes from the top-level call
/// to the lower function.
/// The length of Indexes should match the shape for the local data and each IndexFormula value
/// gives the index on this dimension as offset + sum ids.1 * ids.0
#[derive(Debug, Clone)]
struct Indexes(Vec<IndexFormula>);

impl Indexes {
    fn layout_op(&self, op: &crate::lang::op::LayoutOp, shape: &Shape) -> Result<Self> {
        use crate::lang::op::LayoutOp as L;
        let mut idxs = self.0.clone();
        match op {
            L::Broadcast { broadcasted_dims } => {
                for dim in broadcasted_dims.iter() {
                    if *dim >= idxs.len() {
                        bail!("unexpected dim for broadcast, {dim} {:?}", shape)
                    }
                    idxs[*dim] = IndexFormula { ids: vec![], offset: 0 }
                }
            }
            L::Narrow { dim, offset } => {
                if *dim >= idxs.len() {
                    bail!("unexpected dim for broadcast, {dim} {:?}", shape)
                }
                idxs[*dim].offset += *offset
            }
            L::Transpose { dim1, dim2 } => {
                if *dim1 >= idxs.len() || *dim2 >= idxs.len() {
                    bail!("unexpected dims for transpose {dim1} {dim2}, {:?}", shape)
                }
                idxs.swap(*dim1, *dim2)
            }
            L::SplitDim { dim: _, lhs: _, rhs: _ } | L::MergeDims { dim: _, lhs: _, rhs: _ } => {
                bail!("unsupported layout op {op:?}")
            }
        };
        Ok(Self(idxs))
    }
}

impl From<Id> for IndexFormula {
    fn from(value: Id) -> Self {
        Self { ids: vec![(value, 1)], offset: 0 }
    }
}

impl lang::op::Layout {
    fn lower(&self, idxs: &Indexes) -> Result<(Id, Block)> {
        let strides = self.strides();
        if idxs.0.len() != strides.len() {
            bail!("len mismatch between strides {self:?} and idxs {idxs:?}")
        }
        let off: usize = idxs.0.iter().zip(strides.iter()).map(|(idx, s)| idx.offset * *s).sum();
        let off = (self.offset() + off) as i32;
        let mut acc_id = None;
        let mut block = Block::empty();
        for (idx, &stride) in idxs.0.iter().zip(strides.iter()) {
            for (id, m) in idx.ids.iter() {
                let dim_id = block.mul(*id, (stride * m) as i32);
                let new_id = match acc_id {
                    Some(acc_id) => block.binary(ssa::BinaryOp::Add, dim_id, acc_id, DType::I32),
                    None => block.add(dim_id, off),
                };
                acc_id = Some(new_id)
            }
        }
        match acc_id {
            Some(acc_id) => Ok((acc_id, block)),
            None => {
                let acc_id = Id::new();
                let block = Block::new(vec![(acc_id, SsaI::Const(off.into()))]);
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
        if idxs.0.len() != self.shape().rank() {
            crate::bail!("internal error, idxs {idxs:?}, shape {:?}", self.shape())
        }
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
                let idxs = idxs.layout_op(op, &self.shape)?;
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
