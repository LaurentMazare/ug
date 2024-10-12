#[derive(Debug, Clone, Copy)]
pub struct FlopsMem {
    pub flops: usize,
    pub mem_in_bytes: usize,
}

// Very untyped almost SSA language.
// There are no phi symbols, instead Range is used.
pub mod ssa {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub enum DType {
        PtrF32,
        PtrI32,
        F32,
        I32,
    }

    impl DType {
        pub fn bytes(&self) -> usize {
            match self {
                Self::PtrF32 | Self::PtrI32 => 8,
                Self::F32 | Self::I32 => 4,
            }
        }
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

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub enum Special {
        LocalIdx,
        GridIdx,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Instr {
        DefineAcc(Const),
        DefineGlobal { index: usize, dtype: DType },
        Special(Special),
        Const(Const),
        Unary { op: UnaryOp, arg: VarId, dtype: DType },
        Binary { op: BinaryOp, lhs: VarId, rhs: VarId, dtype: DType },
        Range { lo: VarId, up: VarId, end_idx: usize },
        Load { src: VarId, offset: VarId, dtype: DType },
        Assign { dst: VarId, src: VarId },
        EndRange { start_idx: usize },
        Store { dst: VarId, offset: VarId, value: VarId, dtype: DType },
        Barrier,
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct Kernel {
        pub instrs: Vec<Instr>,
    }

    impl Kernel {
        pub fn flops_mem_per_thread(&self) -> Result<super::FlopsMem> {
            let mut flops = 0usize;
            let mut mem = 0usize;
            let mut mults = vec![];
            let mut mult = 1usize;
            for instr in self.instrs.iter() {
                match instr {
                    Instr::Load { src: _, offset: _, dtype }
                    | Instr::Store { dst: _, offset: _, value: _, dtype } => {
                        mem += mult * dtype.bytes()
                    }
                    Instr::Range { lo, up, end_idx: _ } => {
                        mults.push(mult);
                        let lo = match self.instrs[lo.0] {
                            Instr::Const(Const::I32(lo)) => lo,
                            _ => anyhow::bail!("range lo is not a const"),
                        };
                        let up = match self.instrs[up.0] {
                            Instr::Const(Const::I32(up)) => up,
                            _ => anyhow::bail!("range up is not a const"),
                        };
                        mult *= (up - lo).max(0) as usize;
                    }
                    Instr::EndRange { .. } => match mults.pop() {
                        None => anyhow::bail!("unexpected EndRange"),
                        Some(m) => mult = m,
                    },
                    Instr::Unary { .. } | Instr::Binary { .. } => flops += mult,
                    Instr::DefineGlobal { .. }
                    | Instr::DefineAcc(_)
                    | Instr::Special(_)
                    | Instr::Assign { .. }
                    | Instr::Const(_)
                    | Instr::Barrier => {}
                }
            }
            Ok(super::FlopsMem { flops, mem_in_bytes: mem })
        }
    }

    impl std::fmt::Debug for Kernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for (var_id, instr) in self.instrs.iter().enumerate() {
                writeln!(f, "{var_id:03} {instr:?}")?
            }
            Ok(())
        }
    }
}
