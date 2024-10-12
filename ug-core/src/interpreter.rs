use crate::lang::ssa::{self, Const, Instr, Kernel, VarId};
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

#[derive(Debug, Clone, Copy)]
pub struct W<T, const N: usize>([T; N]);

impl<T: num::traits::Signed + Copy, const N: usize> W<T, N> {
    fn neg(&self) -> Self {
        Self(self.0.map(|v| v.neg()))
    }
}

impl<T: num::traits::real::Real + Copy, const N: usize> W<T, N> {
    fn exp(&self) -> Self {
        Self(self.0.map(|v| v.exp()))
    }
}

impl<T: num::traits::Num + Copy, const N: usize> W<T, N> {
    fn add(&self, rhs: &Self) -> Self {
        W(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
    fn mul(&self, rhs: &Self) -> Self {
        W(std::array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
    fn sub(&self, rhs: &Self) -> Self {
        W(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
    fn div(&self, rhs: &Self) -> Self {
        W(std::array::from_fn(|i| self.0[i] / rhs.0[i]))
    }
}

impl<T: Copy, const N: usize> W<T, N> {
    fn splat(v: T) -> Self {
        W([v; N])
    }
}

use buffer::Id as BufId;

#[derive(Debug, Clone, Copy)]
pub enum Value<const N: usize> {
    Ptr(BufId),
    I32(W<i32, N>),
    F32(W<f32, N>),
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Buffer {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl<const N: usize> Value<N> {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn as_i32(&self) -> Result<&W<i32, N>> {
        if let Self::I32(v) = self {
            Ok(v)
        } else {
            anyhow::bail!("not an i32 {self:?}")
        }
    }

    pub fn as_f32(&self) -> Result<&W<f32, N>> {
        if let Self::F32(v) = self {
            Ok(v)
        } else {
            anyhow::bail!("not a f32 {self:?}")
        }
    }
}

#[derive(Default)]
struct Context<'a, const N: usize> {
    buffers: Vec<&'a mut Buffer>,
    values: Vec<Value<N>>,
}

impl<'a, const N: usize> Context<'a, N> {
    fn new(buffers: Vec<&'a mut Buffer>, n: usize) -> Self {
        Self { buffers, values: vec![Value::None; n] }
    }

    fn set(&mut self, id: VarId, value: Value<N>) -> Result<()> {
        let id = id.as_usize();
        match self.values.get_mut(id) {
            None => anyhow::bail!("set out of bound {id}"),
            Some(dst) => *dst = value,
        }
        Ok(())
    }

    fn get(&mut self, id: VarId) -> Result<Value<N>> {
        let id = id.as_usize();
        match self.values.get(id) {
            None => anyhow::bail!("get out of bound {id:?}"),
            Some(dst) => Ok(*dst),
        }
    }
}

pub fn eval_ssa<const N: usize>(
    kernel: &Kernel,
    buffers: Vec<&mut Buffer>,
    _args: &[Value<N>],
) -> Result<()> {
    let mut context: Context<'_, N> = Context::new(buffers, kernel.instrs.len());
    let mut current_idx = 0;

    while let Some(instr) = kernel.instrs.get(current_idx) {
        let var_id = VarId::new(current_idx);
        let (value, jump_idx) = match instr {
            Instr::DefineGlobal { index, dtype: _ } => (Value::Ptr(BufId::new(*index)), None),
            Instr::Const(Const::F32(v)) => (Value::F32(W::splat(*v)), None),
            Instr::Const(Const::I32(v)) => (Value::I32(W::splat(*v)), None),
            Instr::Range { lo, up, end_idx } => {
                if current_idx >= context.values.len() {
                    anyhow::bail!("get out of bounds {current_idx}")
                }
                if context.values[current_idx].is_none() {
                    (context.get(*lo)?, None)
                } else {
                    let v = context.values[current_idx].as_i32()?.add(&W::splat(1));
                    let up = context.get(*up)?;
                    let up = up.as_i32()?;
                    let mut all_jump = true;
                    let mut any_jump = false;
                    for i in 0..N {
                        if v.0[i] >= up.0[i] {
                            any_jump = true
                        } else {
                            all_jump = false
                        }
                    }
                    if all_jump != any_jump {
                        anyhow::bail!("diverging threads in wrap")
                    }
                    let jump_idx = if all_jump { Some(*end_idx) } else { None };
                    (Value::I32(v), jump_idx)
                }
            }
            Instr::Assign { dst, src } => {
                let value = context.get(*src).unwrap();
                context.set(*dst, value)?;
                (value, None)
            }
            Instr::EndRange { start_idx } => (Value::None, Some(*start_idx)),
            Instr::Load { src, offset, dtype: _ } => {
                let offset = context.get(*offset)?;
                let offset = offset.as_i32()?;
                let value = match context.get(*src)? {
                    Value::Ptr(idx) => match &context.buffers[idx.as_usize()] {
                        Buffer::F32(v) => Value::F32(W(offset.0.map(|o| v[o as usize]))),
                        Buffer::I32(v) => Value::I32(W(offset.0.map(|o| v[o as usize]))),
                    },
                    _ => anyhow::bail!("unexpected dtype for src in load {src:?}"),
                };
                (value, None)
            }
            Instr::Store { dst, offset, value, dtype: _ } => {
                let offset = context.get(*offset)?;
                let offset = offset.as_i32()?;
                let value = context.get(*value)?;
                match context.get(*dst)? {
                    Value::Ptr(idx) => match (context.buffers.get_mut(idx.as_usize()), value) {
                        (Some(Buffer::F32(vs)), Value::F32(v)) => {
                            offset.0.iter().zip(v.0.iter()).for_each(|(o, v)| vs[*o as usize] = *v)
                        }
                        (Some(Buffer::I32(vs)), Value::I32(v)) => {
                            offset.0.iter().zip(v.0.iter()).for_each(|(o, v)| vs[*o as usize] = *v)
                        }
                        (_, v) => anyhow::bail!("unexpected dtype for value in store {v:?}"),
                    },
                    _ => anyhow::bail!("unexpected dtype for src in store {dst:?}"),
                }
                (value, None)
            }
            Instr::Binary { op, lhs, rhs, dtype: _ } => {
                use crate::lang::ssa::BinaryOp as B;
                let lhs = context.get(*lhs)?;
                let rhs = context.get(*rhs)?;
                let v = match (op, &lhs, &rhs) {
                    (B::Add, Value::F32(v1), Value::F32(v2)) => Value::F32(v1.add(v2)),
                    (B::Add, Value::I32(v1), Value::I32(v2)) => Value::I32(v1.add(v2)),
                    (B::Add, _, _) => anyhow::bail!("dtype mismatch for {op:?}"),
                    (B::Mul, Value::F32(v1), Value::F32(v2)) => Value::F32(v1.mul(v2)),
                    (B::Mul, Value::I32(v1), Value::I32(v2)) => Value::I32(v1.mul(v2)),
                    (B::Mul, _, _) => anyhow::bail!("dtype mismatch for {op:?}"),
                    (B::Sub, Value::F32(v1), Value::F32(v2)) => Value::F32(v1.sub(v2)),
                    (B::Sub, Value::I32(v1), Value::I32(v2)) => Value::I32(v1.sub(v2)),
                    (B::Sub, _, _) => anyhow::bail!("dtype mismatch for {op:?}"),
                    (B::Div, Value::F32(v1), Value::F32(v2)) => Value::F32(v1.div(v2)),
                    (B::Div, Value::I32(v1), Value::I32(v2)) => Value::I32(v1.div(v2)),
                    (B::Div, _, _) => anyhow::bail!("dtype mismatch for {op:?}"),
                };
                (v, None)
            }
            Instr::Unary { op, arg, dtype: _ } => {
                use crate::lang::ssa::UnaryOp as U;
                let arg = context.get(*arg)?;
                let v = match (op, &arg) {
                    (U::Neg, Value::F32(v)) => Value::F32(v.neg()),
                    (U::Neg, Value::I32(v)) => Value::I32(v.neg()),
                    (U::Neg, _) => anyhow::bail!("dtype mismatch for {op:?} {arg:?}"),
                    (U::Exp, Value::F32(v)) => Value::F32(v.exp()),
                    (U::Exp, _) => anyhow::bail!("dtype mismatch for {op:?} {arg:?}"),
                };
                (v, None)
            }
            // DefineAcc is handled in the same way as Const, the only difference is that Assign
            // can modify it. This isn't even checked for in this interpreter, all the vars can be
            // assigned but optimizations/code generation might rely on it.
            Instr::DefineAcc(Const::F32(v)) => (Value::F32(W::splat(*v)), None),
            Instr::DefineAcc(Const::I32(v)) => (Value::I32(W::splat(*v)), None),
            Instr::Special(ssa::Special::LocalIdx) => {
                todo!()
            }
            Instr::Special(ssa::Special::GridIdx) => (Value::I32(W::splat(0i32)), None),
            Instr::Barrier => (Value::None, None),
        };
        if !value.is_none() {
            context.set(var_id, value)?;
        }
        current_idx = jump_idx.unwrap_or(current_idx + 1);
    }
    Ok(())
}
