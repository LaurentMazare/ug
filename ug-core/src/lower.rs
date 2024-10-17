use crate::lang::{self, ssa};
use anyhow::{Context, Result};
use ssa::Instr as SsaI;

// ssa::Instr are indexed based on their line number which is not convenient when
// combining blocks of generated instructions
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Id(usize);
impl Id {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub(crate) fn to_varid(self) -> ssa::VarId {
        ssa::VarId::new(self.0)
    }

    pub(crate) fn from_varid(v: ssa::VarId) -> Id {
        Id(v.as_usize())
    }
}

// ssa like instructions but with explicit dst
#[derive(Debug)]
pub(crate) struct Block(pub(crate) Vec<(Id, SsaI)>);

impl Block {
    pub(crate) fn add(self, src_id: Id, v: i32) -> (Id, Self) {
        if v == 0 {
            (src_id, self)
        } else {
            let dst_id = Id::new();
            let mut insts = self.0;
            let cst_id = Id::new();
            insts.push((cst_id, SsaI::Const(v.into())));
            insts.push((
                dst_id,
                SsaI::Binary {
                    op: lang::BinaryOp::Add,
                    lhs: src_id.to_varid(),
                    rhs: cst_id.to_varid(),
                    dtype: lang::DType::I32,
                },
            ));
            (dst_id, Block(insts))
        }
    }

    pub(crate) fn mul(self, src_id: Id, v: i32) -> (Id, Self) {
        if v == 1 {
            (src_id, self)
        } else {
            let dst_id = Id::new();
            let mut insts = self.0;
            let cst_id = Id::new();
            insts.push((cst_id, SsaI::Const(v.into())));
            insts.push((
                dst_id,
                SsaI::Binary {
                    op: lang::BinaryOp::Mul,
                    lhs: src_id.to_varid(),
                    rhs: cst_id.to_varid(),
                    dtype: lang::DType::I32,
                },
            ));
            (dst_id, Block(insts))
        }
    }

    pub(crate) fn empty() -> Self {
        Self(vec![])
    }

    pub(crate) fn new(instrs: Vec<(Id, SsaI)>) -> Self {
        Self(instrs)
    }

    pub(crate) fn relocate(&self) -> Result<Vec<SsaI>> {
        let mut per_id = std::collections::HashMap::new();
        let mut instrs = vec![];
        for (line_idx, (id, instr)) in self.0.iter().enumerate() {
            let line_idx = ssa::VarId::new(line_idx);
            let get_id = |&id| {
                per_id
                    .get(&Id::from_varid(id))
                    .copied()
                    .with_context(|| format!("id not found {id:?}"))
            };
            let instr = match instr {
                SsaI::Store { dst, offset, value, dtype } => {
                    let dst = get_id(dst)?;
                    let offset = get_id(offset)?;
                    let value = get_id(value)?;
                    SsaI::Store { dst, offset, value, dtype: *dtype }
                }
                SsaI::Range { lo, up, end_idx } => {
                    let lo = get_id(lo)?;
                    let up = get_id(up)?;
                    SsaI::Range { lo, up, end_idx: *end_idx }
                }
                SsaI::Load { src, offset, dtype } => {
                    let src = get_id(src)?;
                    let offset = get_id(offset)?;
                    SsaI::Load { src, offset, dtype: *dtype }
                }
                SsaI::Const(c) => SsaI::Const(*c),
                SsaI::Binary { op, lhs, rhs, dtype } => {
                    let lhs = get_id(lhs)?;
                    let rhs = get_id(rhs)?;
                    SsaI::Binary { op: *op, lhs, rhs, dtype: *dtype }
                }
                SsaI::Unary { op, arg, dtype } => {
                    let arg = get_id(arg)?;
                    SsaI::Unary { op: *op, arg, dtype: *dtype }
                }
                SsaI::DefineAcc(c) => SsaI::DefineAcc(*c),
                SsaI::Assign { dst, src } => {
                    let dst = get_id(dst)?;
                    let src = get_id(src)?;
                    SsaI::Assign { dst, src }
                }
                SsaI::Special(s) => SsaI::Special(*s),
                SsaI::DefineLocal { size, dtype } => {
                    SsaI::DefineLocal { size: *size, dtype: *dtype }
                }
                SsaI::DefineGlobal { index, dtype } => {
                    SsaI::DefineGlobal { index: *index, dtype: *dtype }
                }
                SsaI::Barrier => SsaI::Barrier,
                SsaI::EndRange { start_idx } => SsaI::EndRange { start_idx: *start_idx },
            };
            per_id.insert(id, line_idx);
            instrs.push(instr)
        }
        Ok(instrs)
    }
}

impl lang::ExprNode {
    fn lower(
        &self,
        range_id: Id,
        per_arg: &std::collections::HashMap<lang::ArgId, ssa::VarId>,
    ) -> Result<(Id, Block)> {
        use lang::Expr as E;
        let dst_id = Id::new();
        let block = match &self.inner.expr {
            E::Load(src) => {
                let (ptr_i, off_i, src_b) = src.lower(range_id, per_arg)?;
                let instr = SsaI::Load {
                    src: ptr_i,
                    offset: off_i,
                    dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
                };
                let mut src_b = src_b.0;
                src_b.push((dst_id, instr));
                src_b
            }
            E::ScalarConst(c) => {
                let instr = match c {
                    lang::ScalarConst::I32(v) => SsaI::Const(ssa::Const::I32(*v)),
                    lang::ScalarConst::F32(v) => SsaI::Const(ssa::Const::F32(*v)),
                    lang::ScalarConst::Ptr(_) => anyhow::bail!("const ptr are not supported"),
                };
                vec![(dst_id, instr)]
            }
            E::Range(_, _) => anyhow::bail!("TODO range is not supported yet"),
            E::Unary(op, arg) => {
                let (arg_id, arg_b) = arg.lower(range_id, per_arg)?;
                let instr = SsaI::Unary {
                    op: *op,
                    arg: arg_id.to_varid(),
                    dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
                };
                let last = vec![(dst_id, instr)];
                [arg_b.0.as_slice(), last.as_slice()].concat()
            }
            E::Binary(op, lhs, rhs) => {
                let (lhs_id, lhs_b) = lhs.lower(range_id, per_arg)?;
                let (rhs_id, rhs_b) = rhs.lower(range_id, per_arg)?;
                let instr = SsaI::Binary {
                    op: *op,
                    lhs: lhs_id.to_varid(),
                    rhs: rhs_id.to_varid(),
                    dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
                };
                let last = vec![(dst_id, instr)];
                [lhs_b.0.as_slice(), rhs_b.0.as_slice(), last.as_slice()].concat()
            }
        };
        Ok((dst_id, Block(block)))
    }
}

impl lang::StridedSlice {
    fn lower(
        &self,
        range_id: Id,
        per_arg: &std::collections::HashMap<lang::ArgId, ssa::VarId>,
    ) -> Result<(ssa::VarId, ssa::VarId, Block)> {
        let (off_i, off_b) = self.offset().lower()?;
        let (stride_i, stride_b) = self.stride().lower()?;
        let ptr_i = match per_arg.get(&self.ptr().id()) {
            None => anyhow::bail!("unknown arg {:?}", self.ptr().id()),
            Some(id) => *id,
        };
        // TODO(laurent): remove this when we have some proper optimization pass.
        if self.offset().as_const() == Some(0) && self.stride().as_const() == Some(1) {
            Ok((ptr_i, range_id.to_varid(), Block(vec![])))
        } else if self.stride().as_const() == Some(1) {
            let index_i = Id::new();
            let index_b = vec![(
                index_i,
                SsaI::Binary {
                    op: ssa::BinaryOp::Add,
                    lhs: range_id.to_varid(),
                    rhs: off_i.to_varid(),
                    dtype: ssa::DType::I32,
                },
            )];
            let instrs = [off_b.0.as_slice(), stride_b.0.as_slice(), index_b.as_slice()].concat();
            Ok((ptr_i, index_i.to_varid(), Block(instrs)))
        } else {
            let mul_i = Id::new();
            let index_i = Id::new();
            let index_b = vec![
                (
                    mul_i,
                    SsaI::Binary {
                        op: ssa::BinaryOp::Mul,
                        lhs: range_id.to_varid(),
                        rhs: stride_i.to_varid(),
                        dtype: ssa::DType::I32,
                    },
                ),
                (
                    index_i,
                    SsaI::Binary {
                        op: ssa::BinaryOp::Add,
                        lhs: mul_i.to_varid(),
                        rhs: off_i.to_varid(),
                        dtype: ssa::DType::I32,
                    },
                ),
            ];
            let instrs = [off_b.0.as_slice(), stride_b.0.as_slice(), index_b.as_slice()].concat();
            Ok((ptr_i, index_i.to_varid(), Block(instrs)))
        }
    }
}

impl lang::IndexExprNode {
    fn lower(&self) -> Result<(Id, Block)> {
        use lang::IndexExpr as E;
        let dst_id = Id::new();
        let block = match &self.inner.expr {
            E::Add(lhs, rhs) => {
                let (lhs_id, lhs_b) = lhs.lower()?;
                let (rhs_id, rhs_b) = rhs.lower()?;
                let instr = SsaI::Binary {
                    op: ssa::BinaryOp::Add,
                    lhs: lhs_id.to_varid(),
                    rhs: rhs_id.to_varid(),
                    dtype: ssa::DType::I32,
                };
                let last = vec![(dst_id, instr)];
                [lhs_b.0.as_slice(), rhs_b.0.as_slice(), last.as_slice()].concat()
            }
            E::Mul(lhs, rhs) => {
                let (lhs_id, lhs_b) = lhs.lower()?;
                let (rhs_id, rhs_b) = rhs.lower()?;
                let instr = SsaI::Binary {
                    op: ssa::BinaryOp::Mul,
                    lhs: lhs_id.to_varid(),
                    rhs: rhs_id.to_varid(),
                    dtype: ssa::DType::I32,
                };
                let last = vec![(dst_id, instr)];
                [lhs_b.0.as_slice(), rhs_b.0.as_slice(), last.as_slice()].concat()
            }
            E::Const(v) => {
                let instr = SsaI::Const(ssa::Const::I32(*v as i32));
                vec![(dst_id, instr)]
            }
            E::ProgramId => {
                let instr = SsaI::Special(ssa::Special::GridIdx);
                vec![(dst_id, instr)]
            }
        };
        Ok((dst_id, Block(block)))
    }
}

impl lang::Kernel {
    fn lower_b(&self) -> Result<Block> {
        let mut instrs = vec![];
        let mut per_arg = std::collections::HashMap::new();
        for (index, arg) in self.args.iter().enumerate() {
            let id = Id::new();
            let dtype = arg.type_();
            instrs.push((id, SsaI::DefineGlobal { index, dtype }));
            per_arg.insert(arg.id(), id.to_varid());
        }
        for lang::Ops::Store { dst, src } in self.ops.iter() {
            let len = dst.len();

            let (len_i, len_b) = len.lower()?;
            instrs.extend_from_slice(len_b.0.as_slice());
            let lo_id = Id::new();
            instrs.push((lo_id, SsaI::Const(ssa::Const::I32(0))));

            let range_id = Id::new();
            let range = SsaI::Range { lo: lo_id.to_varid(), up: len_i.to_varid(), end_idx: 42 };
            let start_line_idx = instrs.len();
            instrs.push((range_id, range));

            let (src_i, src_b) = src.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(src_b.0.as_slice());
            let (ptr_i, off_i, dst_b) = dst.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(dst_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i,
                value: src_i.to_varid(),
                dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
            };
            instrs.push((Id::new(), store));

            let erange = ssa::Instr::EndRange { start_idx: start_line_idx };
            let end_line_idx = instrs.len();
            instrs.push((Id::new(), erange));
            if let SsaI::Range { end_idx, .. } = &mut instrs[start_line_idx].1 {
                *end_idx = end_line_idx + 1
            }
        }
        Ok(Block(instrs))
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
