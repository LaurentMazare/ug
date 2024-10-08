use crate::lang::ssa::{Const, Expr, Instr, Kernel, VarId};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    PtrI32(Vec<i32>),
    PtrF32(Vec<f32>),
    I32(i32),
    F32(f32),
}

impl Value {
    pub fn as_i32(&self) -> Result<i32> {
        if let Self::I32(v) = self {
            Ok(*v)
        } else {
            anyhow::bail!("not an i32 {self:?}")
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        if let Self::F32(v) = self {
            Ok(*v)
        } else {
            anyhow::bail!("not a f32 {self:?}")
        }
    }
}

#[derive(Default)]
struct Context {
    values: std::collections::HashMap<VarId, Value>,
}

impl Context {
    fn new() -> Self {
        Self { values: std::collections::HashMap::new() }
    }

    fn set(&mut self, id: VarId, value: Value) -> Result<()> {
        if self.values.insert(id, value).is_some() {
            anyhow::bail!("already has value for {id:?}")
        }
        Ok(())
    }

    fn get(&mut self, id: VarId) -> Result<Value> {
        match self.values.get(&id) {
            None => anyhow::bail!("no value for {id:?}"),
            Some(v) => Ok(v.clone()),
        }
    }
}

pub fn eval_ssa(kernel: &Kernel, _args: &[Value]) -> Result<()> {
    let mut context = Context::new();
    let mut current_idx = 0;

    while let Some(instr) = kernel.instrs.get(current_idx) {
        match instr {
            Instr::Affect { var_id, expr, dtype: _ } => {
                let (value, jump_idx) = match expr {
                    Expr::Const(Const::F32(v)) => (Value::F32(*v), None),
                    Expr::Const(Const::I32(v)) => (Value::I32(*v), None),
                    Expr::Range { lo, up, end_idx } => match context.values.get(var_id) {
                        None => (context.get(*lo)?, None),
                        Some(v) => {
                            let v = v.as_i32()? + 1;
                            let up = context.get(*up)?.as_i32()?;
                            let jump_idx = if v < up { None } else { Some(*end_idx) };
                            (Value::I32(v), jump_idx)
                        }
                    },
                    _ => todo!(),
                };
                context.set(*var_id, value)?;
                current_idx = match jump_idx {
                    Some(jump_idx) => jump_idx,
                    None => current_idx + 1,
                };
            }
            Instr::Assign { dst, src } => {
                let value = context.get(*src).unwrap().clone();
                context.set(*dst, value)?;
                current_idx += 1;
            }
            Instr::EndRange { start_idx } => {
                current_idx = *start_idx;
            }
            Instr::Store { .. } => {
                // current_idx += 1;
                todo!();
            }
        }
    }
    Ok(())
}
