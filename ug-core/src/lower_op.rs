use crate::lang::{self, ssa};
use crate::lower::{Block, Id};
use anyhow::Result;
use ssa::{DType, Instr as SsaI};

#[derive(Debug, Clone)]
struct Index {
    id: Id,
    broadcast: bool,
}

#[derive(Debug, Clone)]
struct Indexes(Vec<Index>);

impl lang::op::Layout {
    fn lower(&self, idxs: &Indexes) -> Result<(Id, Block)> {
        let strides = self.strides();
        if idxs.0.len() != strides.len() {
            anyhow::bail!(
                "len mismatch between strides {} and idxs {}",
                strides.len(),
                idxs.0.len()
            )
        }
        let mut acc_id = Id::new();
        let off = self.offset() as i32;
        let mut block = Block::new(vec![(acc_id, SsaI::Const(off.into()))]);
        for (idx, &stride) in idxs.0.iter().zip(strides.iter()) {
            if idx.broadcast {
                continue;
            }
            let dim_id = block.mul(idx.id, stride as i32);
            acc_id = block.binop(ssa::BinaryOp::Add, dim_id, acc_id, DType::I32);
        }
        Ok((acc_id, block))
    }
}

impl lang::op::ReduceOp {
    fn init_value(&self, dtype: DType) -> Result<ssa::Const> {
        let value = match (self, dtype) {
            (Self::Sum, DType::F32) => ssa::Const::F32(0f32),
            (Self::Sum, DType::I32) => ssa::Const::I32(0i32),
            (Self::Prod, DType::F32) => ssa::Const::F32(1f32),
            (Self::Prod, DType::I32) => ssa::Const::I32(1i32),
            (Self::Min, DType::F32) => ssa::Const::F32(f32::MAX),
            (Self::Min, DType::I32) => ssa::Const::I32(i32::MAX),
            (Self::Max, DType::F32) => ssa::Const::F32(f32::MIN),
            (Self::Max, DType::I32) => ssa::Const::I32(i32::MIN),
            (_, DType::PtrF32) | (_, DType::PtrI32) => {
                anyhow::bail!("incorrect dtype for reduce {dtype:?}")
            }
        };
        Ok(value)
    }
    fn fold_op(&self) -> lang::op::BinaryOp {
        match self {
            Self::Sum => lang::BinaryOp::Add,
            Self::Prod => lang::BinaryOp::Mul,
            Self::Max => lang::BinaryOp::Max,
            Self::Min => lang::BinaryOp::Min,
        }
    }
}

impl lang::op::Ast {
    fn lower(
        &self,
        idxs: &Indexes,
        per_arg: &std::collections::HashMap<lang::ArgId, ssa::VarId>,
    ) -> Result<(Id, Block)> {
        use lang::op::AstInner as A;
        let dtype = self.dtype;
        let dst_i = Id::new();
        let instrs = match self.inner.as_ref() {
            A::Load { src, layout } => {
                let ptr_i = match per_arg.get(src) {
                    None => anyhow::bail!("unknown arg {src:?}"),
                    Some(id) => *id,
                };
                let (off_i, off_b) = layout.lower(idxs)?;
                let load = SsaI::Load { src: ptr_i, dtype, offset: off_i.to_varid() };
                let mut off_b = off_b.0;
                off_b.push((dst_i, load));
                off_b
            }
            A::Broadcast { arg: _, axis: _, dim_len: _ } => {
                todo!()
                // arg.lower(idxs, per_arg)
            }
            A::Const(c) => {
                vec![(dst_i, SsaI::Const(*c))]
            }
            A::Unary { op, arg } => {
                let (arg_i, arg_b) = arg.lower(idxs, per_arg)?;
                let mut arg_b = arg_b.0;
                arg_b.push((dst_i, SsaI::Unary { op: *op, arg: arg_i.to_varid(), dtype }));
                arg_b
            }
            A::Reduce { op, arg, axis } => {
                let mut block = Block::empty();
                let init_value = op.init_value(self.dtype)?;
                let fold_op = op.fold_op();
                let define_acc = SsaI::DefineAcc(init_value);
                block.0.push((dst_i, define_acc));
                let reduce_len = match self.shape.dims().get(*axis) {
                    None => anyhow::bail!("unexpected axis for reduce, {axis} {:?}", self.shape),
                    Some(v) => *v,
                };
                let r = block.range(0, reduce_len as i32);

                let mut reduce_idxs = idxs.clone();
                reduce_idxs.0[*axis] = Index { id: r.id(), broadcast: false };
                let (arg_i, arg_b) = arg.lower(&reduce_idxs, per_arg)?;
                block.0.extend_from_slice(&arg_b.0);
                let src_id = Id::new();
                let fold_op = SsaI::Binary {
                    op: fold_op,
                    lhs: dst_i.to_varid(),
                    rhs: arg_i.to_varid(),
                    dtype: self.dtype,
                };
                block.0.push((src_id, fold_op));
                block.0.push((
                    Id::new(),
                    SsaI::Assign { dst: dst_i.to_varid(), src: src_id.to_varid() },
                ));
                block.end_range(r)?;
                block.0
            }
            A::Binary { op, lhs, rhs } => {
                let (lhs_i, lhs_b) = lhs.lower(idxs, per_arg)?;
                let (rhs_i, rhs_b) = rhs.lower(idxs, per_arg)?;
                let op =
                    SsaI::Binary { op: *op, dtype, lhs: lhs_i.to_varid(), rhs: rhs_i.to_varid() };
                [lhs_b.0.as_slice(), rhs_b.0.as_slice(), &[(dst_i, op)]].concat()
            }
        };
        Ok((dst_i, Block(instrs)))
    }
}

impl lang::op::Kernel {
    fn lower_b(&self) -> Result<Block> {
        let mut block = Block::empty();
        let mut per_arg = std::collections::HashMap::new();
        for (index, arg) in self.args.iter().enumerate() {
            let id = Id::new();
            let dtype = arg.type_();
            block.0.push((id, SsaI::DefineGlobal { index, dtype }));
            per_arg.insert(arg.id(), id.to_varid());
        }

        for lang::op::Store { dst, layout, value } in self.ops.iter() {
            let ptr_i = match per_arg.get(dst) {
                None => anyhow::bail!("unknown arg {dst:?}"),
                Some(id) => *id,
            };
            let mut ranges = vec![];
            for &len in layout.dims().iter() {
                let r = block.range(0, len as i32);
                ranges.push(r)
            }
            let idxs =
                Indexes(ranges.iter().map(|v| Index { id: v.id(), broadcast: false }).collect());

            let (off_i, off_b) = layout.lower(&idxs)?;
            block.0.extend_from_slice(off_b.0.as_slice());

            let (src_i, src_b) = value.lower(&idxs, &per_arg)?;
            block.0.extend_from_slice(src_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i.to_varid(),
                value: src_i.to_varid(),
                dtype: value.dtype,
            };
            block.0.push((Id::new(), store));
            for r in ranges.into_iter().rev() {
                block.end_range(r)?;
            }
        }
        Ok(block)
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
