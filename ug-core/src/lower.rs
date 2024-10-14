#![allow(unused)]
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
}

// ssa like instructions but with explicit dst
struct Block(Vec<(Id, SsaI)>);

fn lower_expr(node: &lang::ExprNode) -> Result<(Id, Block)> {
    use lang::Expr as E;
    let dst_id = Id::new();
    let block = match &node.inner.expr {
        E::Load(src) => {
            let ptr = src.ptr();
            let offset = src.offset();
            let len = src.len();
            let stride = src.stride();
            todo!()
        }
        E::ScalarConst(c) => {
            let instr = match c {
                lang::ScalarConst::I32(v) => SsaI::Const(ssa::Const::I32(*v)),
                lang::ScalarConst::F32(v) => SsaI::Const(ssa::Const::F32(*v)),
                lang::ScalarConst::Ptr(_) => anyhow::bail!("const ptr are not supported"),
            };
            vec![(dst_id, instr)]
        }
        E::Range(_, _) => todo!(),
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

pub fn lower(kernel: &lang::Kernel) -> Result<ssa::Kernel> {
    let mut instrs = vec![];
    for (index, arg) in kernel.args.iter().enumerate() {
        let dtype = match arg.type_() {
            lang::ArgType::Ptr => ssa::DType::PtrF32, // TODO(laurent): other pointer types
            lang::ArgType::I32 => ssa::DType::I32,
        };
        instrs.push(ssa::Instr::DefineGlobal { index, dtype })
    }
    for op in kernel.ops.iter() {
        let lang::Ops::Store { dst, src } = op;
        let _ptr = dst.ptr();
        let _offset = dst.offset();
        let len = dst.len().as_const().context("non-const len")?;
        let stride = dst.stride().as_const().context("non-const stride")?;
        let _src = src;
        instrs.push(ssa::Instr::Const(ssa::Const::I32(stride as i32)));
        instrs.push(ssa::Instr::Const(ssa::Const::I32(len as i32)));
        instrs.push(ssa::Instr::Range {
            lo: ssa::VarId::new(0),
            up: ssa::VarId::new(len),
            end_idx: 42,
        });
        instrs.push(ssa::Instr::EndRange { start_idx: 42 });
    }
    Ok(ssa::Kernel { instrs })
}
