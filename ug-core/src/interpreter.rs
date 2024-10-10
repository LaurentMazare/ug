use crate::lang::ssa::{Const, Instr, Kernel, VarId};
use anyhow::Result;
use serde::{Deserialize, Serialize};

mod buffer {
    use serde::{Deserialize, Serialize};
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub struct Id(usize);

    impl Id {
        pub fn new(v: usize) -> Self {
            Self(v)
        }

        pub fn as_usize(&self) -> usize {
            self.0
        }
    }
}

use buffer::Id as BufId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Ptr(BufId),
    I32(i32),
    F32(f32),
    None,
}

pub enum Buffer {
    F32(Vec<f32>),
    I32(Vec<i32>),
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
struct Context<'a> {
    buffers: Vec<&'a mut Buffer>,
    values: Vec<Value>,
}

impl<'a> Context<'a> {
    fn new(buffers: Vec<&'a mut Buffer>, n: usize) -> Self {
        Self { buffers, values: vec![Value::None; n] }
    }

    fn set(&mut self, id: VarId, value: Value) -> Result<()> {
        let id = id.as_usize();
        match self.values.get_mut(id) {
            None => anyhow::bail!("set out of bound {id}"),
            Some(dst) => *dst = value,
        }
        Ok(())
    }

    fn get(&mut self, id: VarId) -> Result<Value> {
        let id = id.as_usize();
        match self.values.get(id) {
            None => anyhow::bail!("get out of bound {id:?}"),
            Some(dst) => Ok(dst.clone()),
        }
    }
}

pub fn eval_ssa(kernel: &Kernel, buffers: Vec<&mut Buffer>, _args: &[Value]) -> Result<()> {
    let mut context = Context::new(buffers, kernel.instrs.len());
    let mut current_idx = 0;

    while let Some(instr) = kernel.instrs.get(current_idx) {
        let var_id = VarId::new(current_idx);
        match instr {
            Instr::DefineGlobal(idx) => context.set(var_id, Value::Ptr(BufId::new(*idx)))?,
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
            Instr::Load { src, offset } => {
                let offset = context.get(*offset)?.as_i32()? as usize;
                match context.get(*src)? {
                    Value::Ptr(idx) => match &context.buffers[idx.as_usize()] {
                        Buffer::F32(v) => context.set(var_id, Value::F32(v[offset]))?,
                        Buffer::I32(v) => context.set(var_id, Value::I32(v[offset]))?,
                    },
                    _ => anyhow::bail!("unexpected dtype for src in load {src:?}"),
                }
            }
            Instr::Store { dst, offset, value } => {
                let offset = context.get(*offset)?.as_i32()? as usize;
                let value = context.get(*value)?;
                match context.get(*dst)? {
                    Value::Ptr(idx) => match (context.buffers.get_mut(idx.as_usize()), value) {
                        (Some(Buffer::F32(vs)), Value::F32(v)) => vs[offset] = v,
                        (Some(Buffer::I32(vs)), Value::I32(v)) => vs[offset] = v,
                        (_, v) => anyhow::bail!("unexpected dtype for value in store {v:?}"),
                    },
                    _ => anyhow::bail!("unexpected dtype for src in store {dst:?}"),
                }
            }
            s => {
                todo!("{s:?}");
            }
        }
        current_idx += 1;
    }
    Ok(())
}
