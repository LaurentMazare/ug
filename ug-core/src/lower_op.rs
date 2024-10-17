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

    // TODO(laurent): remove this.
    fn lower_r(&self, id: Id) -> Result<(Id, Block)> {
        let idxs = Indexes(vec![Index { id, broadcast: false }]);
        self.lower(&idxs)
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
        range_id: Id,
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
                let (off_i, off_b) = layout.lower_r(range_id)?;
                let load = SsaI::Load { src: ptr_i, dtype, offset: off_i.to_varid() };
                let mut off_b = off_b.0;
                off_b.push((dst_i, load));
                off_b
            }
            A::Const(c) => {
                vec![(dst_i, SsaI::Const(*c))]
            }
            A::Unary { op, arg } => {
                let (arg_i, arg_b) = arg.lower(range_id, per_arg)?;
                let mut arg_b = arg_b.0;
                arg_b.push((dst_i, SsaI::Unary { op: *op, arg: arg_i.to_varid(), dtype }));
                arg_b
            }
            A::Reduce { op, arg, axis } => {
                let mut instrs = vec![];
                let init_value = op.init_value(self.dtype)?;
                let fold_op = op.fold_op();
                let define_acc = SsaI::DefineAcc(init_value);
                instrs.push((dst_i, define_acc));
                let lo_id = Id::new();
                instrs.push((lo_id, SsaI::Const(ssa::Const::I32(0))));
                let up_id = Id::new();
                let reduce_len = match self.shape.dims().get(*axis) {
                    None => anyhow::bail!("unexpected axis for reduce, {axis} {:?}", self.shape),
                    Some(v) => *v,
                };
                instrs.push((up_id, SsaI::Const(ssa::Const::I32(reduce_len as i32))));
                let range_id = Id::new();
                let range = SsaI::Range { lo: lo_id.to_varid(), up: up_id.to_varid(), end_idx: 3 };
                let start_line_idx = instrs.len();
                instrs.push((range_id, range));

                // TODO(laurent): pass both the global range_id and the local one.
                let (arg_i, arg_b) = arg.lower(range_id, per_arg)?;
                instrs.extend_from_slice(&arg_b.0);
                let src_id = Id::new();
                let fold_op = SsaI::Binary {
                    op: fold_op,
                    lhs: dst_i.to_varid(),
                    rhs: arg_i.to_varid(),
                    dtype: self.dtype,
                };
                instrs.push((src_id, fold_op));
                instrs.push((
                    Id::new(),
                    SsaI::Assign { dst: dst_i.to_varid(), src: src_id.to_varid() },
                ));

                let erange = ssa::Instr::EndRange { start_idx: 1 };
                let end_line_idx = instrs.len();
                instrs.push((Id::new(), erange));
                if let SsaI::Range { end_idx, .. } = &mut instrs[start_line_idx].1 {
                    *end_idx = end_line_idx + 1
                }
                instrs
            }
            A::Binary { op, lhs, rhs } => {
                let (lhs_i, lhs_b) = lhs.lower(range_id, per_arg)?;
                let (rhs_i, rhs_b) = rhs.lower(range_id, per_arg)?;
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
        let mut instrs = vec![];
        let mut per_arg = std::collections::HashMap::new();
        for (index, arg) in self.args.iter().enumerate() {
            let id = Id::new();
            let dtype = arg.type_();
            instrs.push((id, SsaI::DefineGlobal { index, dtype }));
            per_arg.insert(arg.id(), id.to_varid());
        }

        for lang::op::Store { dst, layout, value } in self.ops.iter() {
            let ptr_i = match per_arg.get(dst) {
                None => anyhow::bail!("unknown arg {dst:?}"),
                Some(id) => *id,
            };
            let len_i = Id::new();
            let num_el = layout.num_elements();
            instrs.push((len_i, SsaI::Const(ssa::Const::I32(num_el as i32))));
            let lo_i = Id::new();
            instrs.push((lo_i, SsaI::Const(ssa::Const::I32(0))));

            let range_id = Id::new();
            let range = SsaI::Range { lo: lo_i.to_varid(), up: len_i.to_varid(), end_idx: 42 };
            let start_line_idx = instrs.len();
            instrs.push((range_id, range));

            let (off_i, off_b) = layout.lower_r(range_id)?;
            instrs.extend_from_slice(off_b.0.as_slice());

            let (src_i, src_b) = value.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(src_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i.to_varid(),
                value: src_i.to_varid(),
                dtype: value.dtype,
            };
            instrs.push((Id::new(), store));

            let erange = ssa::Instr::EndRange { start_idx: start_line_idx };
            let end_line_idx = instrs.len();
            instrs.push((Id::new(), erange));
            if let SsaI::Range { end_idx, .. } = &mut instrs[start_line_idx].1 {
                *end_idx = end_line_idx + 1
            }
        }
        Ok(Block::new(instrs))
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
