use crate::lang::{self, ssa};
use anyhow::{Context, Result};
use ssa::Instr as SsaI;

// ssa::Instr are indexed based on their line number which is not convenient when
// combining blocks of generated instructions
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Id(usize);
impl Id {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    fn to_varid(self) -> ssa::VarId {
        ssa::VarId::new(self.0)
    }

    fn from_varid(v: ssa::VarId) -> Id {
        Id(v.as_usize())
    }
}

// ssa like instructions but with explicit dst
#[derive(Debug)]
struct Block(Vec<(Id, SsaI)>);

impl Block {
    fn relocate(&self) -> Result<Vec<SsaI>> {
        let mut per_id = std::collections::HashMap::new();
        let mut instrs = vec![];
        for (line_idx, (id, instr)) in self.0.iter().enumerate() {
            let line_idx = ssa::VarId::new(line_idx);
            let get_id = |&id| per_id.get(&Id::from_varid(id)).copied().context("id not found");
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

fn lower_expr(node: &lang::ExprNode) -> Result<(Id, Block)> {
    use lang::Expr as E;
    let dst_id = Id::new();
    let block = match &node.inner.expr {
        E::Load(src) => {
            let _ptr = src.ptr();
            let _offset = src.offset();
            let _len = src.len();
            let _stride = src.stride();
            anyhow::bail!("TODO: load is not supported")
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
            let (arg_id, arg_b) = lower_expr(arg)?;
            let instr = SsaI::Unary {
                op: *op,
                arg: arg_id.to_varid(),
                dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
            };
            let last = vec![(dst_id, instr)];
            [arg_b.0.as_slice(), last.as_slice()].concat()
        }
        E::Binary(op, lhs, rhs) => {
            let (lhs_id, lhs_b) = lower_expr(lhs)?;
            let (rhs_id, rhs_b) = lower_expr(rhs)?;
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

fn lower_index(index: &lang::IndexExprNode) -> Result<(Id, Block)> {
    use lang::IndexExpr as E;
    let dst_id = Id::new();
    let block = match &index.inner.expr {
        E::Add(lhs, rhs) => {
            let (lhs_id, lhs_b) = lower_index(lhs)?;
            let (rhs_id, rhs_b) = lower_index(rhs)?;
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
            let (lhs_id, lhs_b) = lower_index(lhs)?;
            let (rhs_id, rhs_b) = lower_index(rhs)?;
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

fn lower_b(kernel: &lang::Kernel) -> Result<Block> {
    let mut instrs = vec![];
    let mut per_arg = std::collections::HashMap::new();
    for (index, arg) in kernel.args.iter().enumerate() {
        let id = Id::new();
        let dtype = match arg.type_() {
            lang::ArgType::Ptr => ssa::DType::PtrF32, // TODO(laurent): support other pointer types
            lang::ArgType::I32 => ssa::DType::I32,
        };
        instrs.push((id, SsaI::DefineGlobal { index, dtype }));
        per_arg.insert(arg.id(), id.to_varid());
    }
    for op in kernel.ops.iter() {
        let lang::Ops::Store { dst, src } = op;
        let offset = dst.offset();
        let len = dst.len();
        let stride = dst.stride();
        let dst = match per_arg.get(&dst.ptr().id()) {
            None => anyhow::bail!("unknown arg {:?}", dst.ptr().id()),
            Some(id) => *id,
        };

        let (off_i, off_b) = lower_index(offset)?;
        instrs.extend_from_slice(off_b.0.as_slice());
        let (len_i, len_b) = lower_index(len)?;
        instrs.extend_from_slice(len_b.0.as_slice());
        let (_stride_i, stride_b) = lower_index(stride)?;
        instrs.extend_from_slice(stride_b.0.as_slice());

        let range_id = Id::new();
        let range = SsaI::Range { lo: ssa::VarId::new(0), up: len_i.to_varid(), end_idx: 42 };
        instrs.push((range_id, range));

        // TODO(laurent): this should take as input the loop index
        let (src_i, src_b) = lower_expr(src)?;
        instrs.extend_from_slice(src_b.0.as_slice());
        let store = SsaI::Store {
            dst,
            // TODO(laurent): compute the offset based on the range idx and stride.
            offset: off_i.to_varid(),
            value: src_i.to_varid(),
            dtype: ssa::DType::F32, // TODO(laurent): support other dtypes
        };
        instrs.push((Id::new(), store));

        let erange = ssa::Instr::EndRange { start_idx: 42 };
        instrs.push((Id::new(), erange));
    }
    Ok(Block(instrs))
}

pub fn lower(kernel: &lang::Kernel) -> Result<ssa::Kernel> {
    let block = lower_b(kernel)?;
    let instrs = block.relocate()?;
    Ok(ssa::Kernel { instrs })
}
