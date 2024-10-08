use crate::lang::ssa::{Const, Instr, Kernel, VarId};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    PtrI32(Vec<i32>),
    PtrF32(Vec<f32>),
    I32(i32),
    F32(f32),
    None,
}

impl Value {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

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
    values: Vec<Value>,
}

impl Context {
    fn new(n: usize) -> Self {
        Self { values: vec![Value::None; n] }
    }

    fn set(&mut self, id: VarId, value: Value) -> Result<()> {
        let id = id.0;
        match self.values.get_mut(id) {
            None => anyhow::bail!("set out of bound {id}"),
            Some(dst) => *dst = value,
        }
        Ok(())
    }

    fn get(&mut self, id: VarId) -> Result<Value> {
        let id = id.0;
        match self.values.get(id) {
            None => anyhow::bail!("get out of bound {id:?}"),
            Some(dst) => Ok(dst.clone()),
        }
    }
}

pub fn eval_ssa(kernel: &Kernel, _args: &[Value]) -> Result<()> {
    let mut context = Context::new(kernel.instrs.len());
    let mut current_idx = 0;

    while let Some(instr) = kernel.instrs.get(current_idx) {
        let var_id = VarId(current_idx);
        match instr {
            Instr::Const(Const::F32(v)) => context.set(var_id, Value::F32(*v))?,
            Instr::Const(Const::I32(v)) => context.set(var_id, Value::I32(*v))?,
            Instr::Range { lo, up, end_idx } => {
                if current_idx >= context.values.len() {
                    anyhow::bail!("get out of bounds {current_idx}")
                }
                if context.values[current_idx].is_none() {
                    context.values[current_idx] = context.get(*lo)?
                } else {
                    let v = context.values[current_idx].as_i32()? + 1;
                    let up = context.get(*up)?.as_i32()?;
                    context.values[current_idx] = Value::I32(v);
                    if v >= up {
                        current_idx = *end_idx;
                        continue;
                    }
                }
            }
            Instr::Assign { dst, src } => {
                let value = context.get(*src).unwrap().clone();
                context.set(*dst, value)?;
            }
            Instr::EndRange { start_idx } => {
                current_idx = *start_idx;
                continue;
            }
            _ => {
                todo!();
            }
        }
        current_idx += 1;
    }
    Ok(())
}
