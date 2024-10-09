pub mod ssa {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum DType {
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

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Const {
        I32(i32),
        F32(f32),
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Instr {
        DefineAcc,
        DefineGlobal,
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
        pub args: Vec<VarId>,
        pub instrs: Vec<Instr>,
    }
}
