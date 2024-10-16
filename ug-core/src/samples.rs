pub mod ssa {
    use crate::lang::ssa::{BinaryOp, Const, DType, Instr as I, Kernel, VarId};

    pub fn simple_add(vec_len: usize) -> Kernel {
        let v = VarId::new;
        let dtype = DType::I32;
        let instrs = vec![
            /* 0 */ I::DefineGlobal { index: 0, dtype: DType::PtrI32 },
            /* 1 */ I::DefineGlobal { index: 1, dtype: DType::PtrI32 },
            /* 2 */ I::DefineGlobal { index: 2, dtype: DType::PtrI32 },
            /* 3 */ I::Const(Const::I32(0)),
            /* 4 */ I::Const(Const::I32(vec_len as i32)),
            /* 5 */ I::Range { lo: v(3), up: v(4), end_idx: 11 },
            /* 6 */ I::Load { src: v(1), offset: v(5), dtype },
            /* 7 */ I::Load { src: v(2), offset: v(5), dtype },
            /* 8 */ I::Binary { op: self::BinaryOp::Add, lhs: v(6), rhs: v(7), dtype },
            /* 9 */ I::Store { dst: v(0), offset: v(5), value: v(8), dtype },
            /* 10 */ I::EndRange { start_idx: 5 },
        ];
        Kernel { instrs }
    }

    pub fn simple_dotprod(vec_len: usize) -> Kernel {
        let v = VarId::new;
        let dtype = DType::F32;
        let instrs = vec![
            /* 0 */ I::DefineGlobal { index: 0, dtype: DType::PtrF32 },
            /* 1 */ I::DefineGlobal { index: 1, dtype: DType::PtrF32 },
            /* 2 */ I::DefineGlobal { index: 2, dtype: DType::PtrF32 },
            /* 3 */ I::Const(Const::I32(0)),
            /* 4 */ I::Const(Const::I32(vec_len as i32)),
            /* 5 */ I::DefineAcc(Const::F32(0.)),
            /* 6 */ I::Range { lo: v(3), up: v(4), end_idx: 13 },
            /* 7 */ I::Load { src: v(1), offset: v(6), dtype },
            /* 8 */ I::Load { src: v(2), offset: v(6), dtype },
            /* 9 */ I::Binary { op: self::BinaryOp::Mul, lhs: v(7), rhs: v(8), dtype },
            /* 10*/ I::Binary { op: self::BinaryOp::Add, lhs: v(9), rhs: v(5), dtype },
            /* 11*/ I::Assign { dst: v(5), src: v(10) },
            /* 12*/ I::EndRange { start_idx: 6 },
            /* 13*/ I::Store { dst: v(0), offset: v(3), value: v(5), dtype },
        ];
        Kernel { instrs }
    }
}

pub mod op {
    // use crate::lang::op::{AstInner as I, BinaryOp, Const, DType, Kernel, VarId};
}

use crate::lang::{Arg, ArgType, ExprNode as E, IndexExprNode as I, Kernel, Ops};

pub fn simple_add(block_size: usize) -> Kernel {
    let lhs_ptr = Arg::new(ArgType::Ptr);
    let rhs_ptr = Arg::new(ArgType::Ptr);
    let dst_ptr = Arg::new(ArgType::Ptr);
    let offset = I::mul(&I::program_id(), &I::cst(block_size));
    let stride = I::cst(1);
    let len = I::cst(block_size);
    let lhs = E::load(&lhs_ptr, &offset, &len, &stride);
    let rhs = E::load(&rhs_ptr, &offset, &len, &stride);
    let op = Ops::store(&dst_ptr, &offset, &len, &stride, &lhs.add(&rhs));
    Kernel::new(format!("simple_add_{block_size}"), vec![lhs_ptr, rhs_ptr, dst_ptr], vec![op])
}
