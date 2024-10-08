#![allow(unused)]
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub mod ssa {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum DType {
        F32,
        I32,
    }

    /// Unique identifier for variables.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub struct VarId(usize);

    impl VarId {
        pub fn new() -> Self {
            // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
            use std::sync::atomic;
            static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
            Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
        }

        pub fn as_usize(&self) -> usize {
            self.0
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum UnaryOp {
        Exp,
        Neg,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum BinaryOp {
        Add,
        Sub,
        Mul,
        Div,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Const {
        I32(i32),
        F32(f32),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Expr {
        DefineAcc,
        DefineGlobal,
        Const(Const),
        Unary { op: UnaryOp, arg: VarId },
        Binary { op: BinaryOp, lhs: VarId, rhs: VarId },
        Range { lo: VarId, up: VarId, end_idx: usize },
        Load { src: VarId, offset: VarId },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Instr {
        Affect { var_id: VarId, expr: Expr, dtype: DType },
        Assign { dst: VarId, src: VarId },
        EndRange { start_idx: usize },
        Store { dst: VarId, offset: VarId, value: VarId },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Kernel {
        pub args: Vec<VarId>,
        pub instrs: Vec<Instr>,
    }

    pub fn const_i32(v: i32) -> (VarId, Instr) {
        let var_id = VarId::new();
        let affect = Instr::Affect { var_id, expr: Expr::Const(Const::I32(v)), dtype: DType::I32 };
        (var_id, affect)
    }

    pub fn const_f32(v: f32) -> (VarId, Instr) {
        let var_id = VarId::new();
        let affect = Instr::Affect { var_id, expr: Expr::Const(Const::F32(v)), dtype: DType::F32 };
        (var_id, affect)
    }
}
