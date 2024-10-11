// Very untyped almost SSA language.
// There are no phi symbols, instead Range is used.
pub mod ssa {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum DType {
        PtrF32,
        PtrI32,
        F32,
        I32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub struct VarId(usize);

    impl VarId {
        pub fn new(v: usize) -> Self {
            Self(v)
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

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub enum Const {
        I32(i32),
        F32(f32),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Instr {
        DefineAcc(Const),
        DefineGlobal { index: usize, dtype: DType },
        Const(Const),
        Unary { op: UnaryOp, arg: VarId },
        Binary { op: BinaryOp, lhs: VarId, rhs: VarId },
        Range { lo: VarId, up: VarId, end_idx: usize },
        Load { src: VarId, offset: VarId },
        Assign { dst: VarId, src: VarId },
        EndRange { start_idx: usize },
        Store { dst: VarId, offset: VarId, value: VarId },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Kernel {
        pub instrs: Vec<Instr>,
    }
}
