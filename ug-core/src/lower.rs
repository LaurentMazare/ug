use crate::lang::{self, ssa};
use crate::Result;
use ssa::Instr as SsaI;

// ssa::Instr are indexed based on their line number which is not convenient when
// combining blocks of generated instructions
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Id(usize);
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

    pub(crate) fn to_a(self) -> ssa::A {
        ssa::A::Var(ssa::VarId::new(self.0))
    }

    pub(crate) fn from_varid(v: ssa::VarId) -> Id {
        Id(v.as_usize())
    }
}

impl From<Id> for ssa::A {
    fn from(val: Id) -> Self {
        val.to_a()
    }
}

// ssa like instructions but with explicit dst
#[derive(Debug)]
pub(crate) struct Block(pub(crate) Vec<(Id, SsaI)>);

#[derive(Debug)]
pub(crate) struct Range {
    range_id: Id,
    erange_id: Id,
}

impl Range {
    pub(crate) fn id(&self) -> Id {
        self.range_id
    }
}

impl Block {
    pub(crate) fn range(&mut self, lo: i32, up: i32) -> Range {
        let (range_id, erange_id) = (Id::new(), Id::new());
        let range = SsaI::Range { lo: lo.into(), up: up.into(), end_idx: erange_id.to_varid() };
        self.0.push((range_id, range));
        Range { range_id, erange_id }
    }

    pub(crate) fn end_range(&mut self, range: Range) -> Result<()> {
        let erange = ssa::Instr::EndRange { start_idx: range.range_id.to_varid() };
        self.0.push((range.erange_id, erange));
        Ok(())
    }

    #[allow(unused)]
    pub(crate) fn add(&mut self, src_id: Id, v: i32) -> Id {
        if v == 0 {
            src_id
        } else {
            let dst_id = Id::new();
            self.0.push((
                dst_id,
                SsaI::Binary {
                    op: lang::BinaryOp::Add,
                    lhs: src_id.to_a(),
                    rhs: v.into(),
                    dtype: lang::DType::I32,
                },
            ));
            dst_id
        }
    }

    pub(crate) fn mul(&mut self, src_id: Id, v: i32) -> Id {
        if v == 1 {
            src_id
        } else {
            let dst_id = Id::new();
            self.0.push((
                dst_id,
                SsaI::Binary {
                    op: lang::BinaryOp::Mul,
                    lhs: src_id.to_a(),
                    rhs: v.into(),
                    dtype: lang::DType::I32,
                },
            ));
            dst_id
        }
    }

    pub(crate) fn push(&mut self, inst: SsaI) -> Id {
        let id = Id::new();
        self.0.push((id, inst));
        id
    }

    pub(crate) fn unary<I: Into<ssa::A>>(
        &mut self,
        op: lang::UnaryOp,
        arg: I,
        dtype: lang::DType,
    ) -> Id {
        let id = Id::new();
        let op = SsaI::Unary { op, arg: arg.into(), dtype };
        self.0.push((id, op));
        id
    }

    pub(crate) fn binary<I1: Into<ssa::A>, I2: Into<ssa::A>>(
        &mut self,
        op: lang::BinaryOp,
        lhs: I1,
        rhs: I2,
        dtype: lang::DType,
    ) -> Id {
        let id = Id::new();
        let op = SsaI::Binary { op, lhs: lhs.into(), rhs: rhs.into(), dtype };
        self.0.push((id, op));
        id
    }

    #[allow(unused)]
    pub(crate) fn empty() -> Self {
        Self(vec![])
    }

    pub(crate) fn new(instrs: Vec<(Id, SsaI)>) -> Self {
        Self(instrs)
    }

    // This switches all the VarId to be in line number rather than "random" unique identifiers.
    pub(crate) fn relocate(&self) -> Result<Vec<SsaI>> {
        let mut per_id = std::collections::HashMap::new();
        for (line_idx, (id, _)) in self.0.iter().enumerate() {
            let line_idx = ssa::VarId::new(line_idx);
            per_id.insert(id, line_idx);
        }
        let mut instrs = vec![];
        for (_, instr) in self.0.iter() {
            let get_id = |id: ssa::VarId| match per_id.get(&Id::from_varid(id)) {
                Some(v) => Ok(*v),
                None => crate::bail!("id not found {id:?}"),
            };
            let get_a = |a: ssa::A| {
                let a = match a {
                    ssa::A::Var(v) => ssa::A::Var(get_id(v)?),
                    ssa::A::Const(c) => ssa::A::Const(c),
                };
                Ok::<_, crate::Error>(a)
            };
            let instr = match instr {
                SsaI::Store { dst, offset, value, dtype } => {
                    let dst = get_id(*dst)?;
                    let offset = get_a(*offset)?;
                    let value = get_a(*value)?;
                    SsaI::Store { dst, offset, value, dtype: *dtype }
                }
                SsaI::If { cond, end_idx } => {
                    let cond = get_a(*cond)?;
                    let end_idx = get_id(*end_idx)?;
                    SsaI::If { cond, end_idx }
                }
                SsaI::Range { lo, up, end_idx } => {
                    let lo = get_a(*lo)?;
                    let up = get_a(*up)?;
                    let end_idx = get_id(*end_idx)?;
                    SsaI::Range { lo, up, end_idx }
                }
                SsaI::Load { src, offset, dtype } => {
                    let src = get_id(*src)?;
                    let offset = get_a(*offset)?;
                    SsaI::Load { src, offset, dtype: *dtype }
                }
                SsaI::Const(c) => SsaI::Const(*c),
                SsaI::Binary { op, lhs, rhs, dtype } => {
                    let lhs = get_a(*lhs)?;
                    let rhs = get_a(*rhs)?;
                    SsaI::Binary { op: *op, lhs, rhs, dtype: *dtype }
                }
                SsaI::Unary { op, arg, dtype } => {
                    let arg = get_a(*arg)?;
                    SsaI::Unary { op: *op, arg, dtype: *dtype }
                }
                SsaI::DefineAcc(c) => SsaI::DefineAcc(*c),
                SsaI::Assign { dst, src } => {
                    let dst = get_id(*dst)?;
                    let src = get_a(*src)?;
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
                SsaI::ReduceLocal { op, arg, dtype } => {
                    let arg = get_a(*arg)?;
                    SsaI::ReduceLocal { op: *op, arg, dtype: *dtype }
                }
                SsaI::EndIf => SsaI::EndIf,
                SsaI::EndRange { start_idx } => {
                    let start_idx = get_id(*start_idx)?;
                    SsaI::EndRange { start_idx }
                }
            };
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
        let dtype = self.dtype();
        let block = match &self.inner.expr {
            E::Load(src) => {
                let (ptr_i, off_i, src_b) = src.lower(range_id, per_arg)?;
                let instr = SsaI::Load { src: ptr_i, offset: off_i.into(), dtype };
                let mut src_b = src_b.0;
                src_b.push((dst_id, instr));
                src_b
            }
            E::Const(c) => vec![(dst_id, SsaI::Const(*c))],
            E::Range(_, _) => crate::bail!("TODO range is not supported yet"),
            E::Unary(op, arg) => {
                let (arg_id, arg_b) = arg.lower(range_id, per_arg)?;
                let instr = SsaI::Unary { op: *op, arg: arg_id.to_a(), dtype };
                let last = vec![(dst_id, instr)];
                [arg_b.0.as_slice(), last.as_slice()].concat()
            }
            E::Binary(op, lhs, rhs) => {
                let (lhs_id, lhs_b) = lhs.lower(range_id, per_arg)?;
                let (rhs_id, rhs_b) = rhs.lower(range_id, per_arg)?;
                let instr = SsaI::Binary { op: *op, lhs: lhs_id.to_a(), rhs: rhs_id.to_a(), dtype };
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
            None => crate::bail!("unknown arg {:?}", self.ptr().id()),
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
                    lhs: range_id.to_a(),
                    rhs: off_i.to_a(),
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
                        lhs: range_id.to_a(),
                        rhs: stride_i.to_a(),
                        dtype: ssa::DType::I32,
                    },
                ),
                (
                    index_i,
                    SsaI::Binary {
                        op: ssa::BinaryOp::Add,
                        lhs: mul_i.to_a(),
                        rhs: off_i.to_a(),
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
                    lhs: lhs_id.to_a(),
                    rhs: rhs_id.to_a(),
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
                    lhs: lhs_id.to_a(),
                    rhs: rhs_id.to_a(),
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
            let dtype = match arg.type_() {
                ssa::Type::Ptr(v) => v,
                ssa::Type::Value(_) => crate::bail!("non-pointer arguments are not supported yet"),
            };
            instrs.push((id, SsaI::DefineGlobal { index, dtype }));
            per_arg.insert(arg.id(), id.to_varid());
        }
        for lang::Ops::Store { dst, src } in self.ops.iter() {
            let len = dst.len();

            let (len_i, len_b) = len.lower()?;
            instrs.extend_from_slice(len_b.0.as_slice());
            let lo_id = Id::new();
            instrs.push((lo_id, SsaI::Const(ssa::Const::I32(0))));

            let (range_id, erange_id) = (Id::new(), Id::new());
            let range =
                SsaI::Range { lo: lo_id.to_a(), up: len_i.to_a(), end_idx: erange_id.to_varid() };
            instrs.push((range_id, range));

            let (src_i, src_b) = src.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(src_b.0.as_slice());
            let (ptr_i, off_i, dst_b) = dst.lower(range_id, &per_arg)?;
            instrs.extend_from_slice(dst_b.0.as_slice());
            let store = SsaI::Store {
                dst: ptr_i,
                offset: off_i.into(),
                value: src_i.to_a(),
                dtype: src.dtype(),
            };
            instrs.push((Id::new(), store));

            let erange = ssa::Instr::EndRange { start_idx: range_id.to_varid() };
            instrs.push((erange_id, erange));
        }
        Ok(Block(instrs))
    }

    pub fn lower(&self) -> Result<ssa::Kernel> {
        let block = self.lower_b()?;
        let instrs = block.relocate()?;
        Ok(ssa::Kernel { instrs })
    }
}
