pub mod ssa {
    use crate::lang::ssa;
    use crate::lang::ssa::{BinaryOp, Const, DType, Instr as I, Kernel, VarId, A};
    use anyhow::Result;

    pub fn simple_add(vec_len: usize) -> Kernel {
        let v = VarId::new;
        let a = |i| A::Var(VarId::new(i));
        let i32 = |i| A::Const(Const::I32(i));
        let dtype = DType::I32;
        let instrs = vec![
            /* 0 */ I::DefineGlobal { index: 0, dtype: DType::PtrI32 },
            /* 1 */ I::DefineGlobal { index: 1, dtype: DType::PtrI32 },
            /* 2 */ I::DefineGlobal { index: 2, dtype: DType::PtrI32 },
            /* 3 */ I::Range { lo: i32(0), up: i32(vec_len as i32), end_idx: v(8) },
            /* 4 */ I::Load { src: v(1), offset: a(3), dtype },
            /* 5 */ I::Load { src: v(2), offset: a(3), dtype },
            /* 6 */ I::Binary { op: self::BinaryOp::Add, lhs: a(4), rhs: a(5), dtype },
            /* 7 */ I::Store { dst: v(0), offset: a(3), value: a(6), dtype },
            /* 8 */ I::EndRange { start_idx: v(3) },
        ];
        Kernel { instrs }
    }

    pub fn simple_dotprod(vec_len: usize) -> Kernel {
        let v = VarId::new;
        let a = |i| A::Var(VarId::new(i));
        let dtype = DType::F32;
        let instrs = vec![
            /* 0 */ I::DefineGlobal { index: 0, dtype: DType::PtrF32 },
            /* 1 */ I::DefineGlobal { index: 1, dtype: DType::PtrF32 },
            /* 2 */ I::DefineGlobal { index: 2, dtype: DType::PtrF32 },
            /* 3 */ I::Const(Const::I32(0)),
            /* 4 */ I::Const(Const::I32(vec_len as i32)),
            /* 5 */ I::DefineAcc(Const::F32(0.)),
            /* 6 */ I::Range { lo: a(3), up: a(4), end_idx: v(12) },
            /* 7 */ I::Load { src: v(1), offset: a(6), dtype },
            /* 8 */ I::Load { src: v(2), offset: a(6), dtype },
            /* 9 */ I::Binary { op: self::BinaryOp::Mul, lhs: a(7), rhs: a(8), dtype },
            /* 10*/ I::Binary { op: self::BinaryOp::Add, lhs: a(9), rhs: a(5), dtype },
            /* 11*/ I::Assign { dst: v(5), src: a(10) },
            /* 12*/ I::EndRange { start_idx: v(6) },
            /* 13*/ I::Store { dst: v(0), offset: a(3), value: a(5), dtype },
        ];
        Kernel { instrs }
    }

    pub fn softmax(_dim1: usize, dim2: usize) -> Result<Kernel> {
        let mut b = crate::lower::Block::empty();
        let dtype = DType::F32;
        let src_i = b.push(I::DefineGlobal { index: 0, dtype: DType::PtrF32 });
        let dst_i = b.push(I::DefineGlobal { index: 1, dtype: DType::PtrF32 });
        let g_i = b.push(I::Special(ssa::Special::GridIdx));
        let l_i = b.push(I::Special(ssa::Special::LocalIdx));
        let off_i = b.mul(g_i, dim2 as i32);
        let off_i = b.binop(BinaryOp::Add, off_i, l_i, dtype);
        let load_i = b.push(I::Load { src: src_i.to_varid(), offset: off_i.to_a(), dtype });
        // TODO: Compute the max value over dim2
        let value_i = b.unary(ssa::UnaryOp::Exp, load_i, dtype);
        // TODO: Normalize
        b.push(I::Store {
            dst: dst_i.to_varid(),
            offset: off_i.to_a(),
            value: value_i.to_a(),
            dtype,
        });
        Ok(Kernel { instrs: b.relocate()? })
    }
}

pub mod op {
    use crate::lang::op::{self, Arg, DType, Kernel, Layout};
    use anyhow::Result;

    pub fn softmax(dim1: usize, dim2: usize) -> Result<Kernel> {
        let layout = Layout::from_shape(&[dim1, dim2]);
        let src_ptr = Arg::new(DType::PtrF32);
        let dst_ptr = Arg::new(DType::PtrF32);
        let src = op::load(src_ptr.id(), layout.clone(), DType::F32)?;
        let src_max = op::reduce(op::ReduceOp::Max, src.clone(), 1)?;
        let src_max = op::broadcast(src_max, 1, dim2)?;
        let diff = op::binary(op::BinaryOp::Sub, src, src_max)?;
        let exp = op::unary(op::UnaryOp::Exp, diff)?;
        let sum_exp = op::reduce(op::ReduceOp::Sum, exp.clone(), 1)?;
        let sum_exp = op::broadcast(sum_exp, 1, dim2)?;
        let sm = op::binary(op::BinaryOp::Div, exp, sum_exp)?;
        let st = op::store(dst_ptr.id(), layout, sm)?;
        let kernel =
            Kernel::new(format!("softmax_{dim1}_{dim2}"), vec![src_ptr, dst_ptr], vec![st]);
        Ok(kernel)
    }
}

use crate::lang::{Arg, DType, ExprNode as E, IndexExprNode as I, Kernel, Ops};

pub fn simple_add(block_size: usize) -> Kernel {
    let lhs_ptr = Arg::new(DType::PtrF32);
    let rhs_ptr = Arg::new(DType::PtrF32);
    let dst_ptr = Arg::new(DType::PtrF32);
    let offset = I::mul(&I::program_id(), &I::cst(block_size));
    let stride = I::cst(1);
    let len = I::cst(block_size);
    let lhs = E::load(&lhs_ptr, &offset, &len, &stride);
    let rhs = E::load(&rhs_ptr, &offset, &len, &stride);
    let op = Ops::store(&dst_ptr, &offset, &len, &stride, &lhs.add(&rhs));
    Kernel::new(format!("simple_add_{block_size}"), vec![lhs_ptr, rhs_ptr, dst_ptr], vec![op])
}
