//! Compilation cache utilities.
use crate::lang::op::{self, ArgId, Ast};
use crate::Result;
use std::collections::HashMap;

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct NormalizedKernel {
    pub(crate) args: Vec<op::Arg>,
    pub(crate) ops: Vec<op::Store>,
}

impl NormalizedKernel {
    pub fn new(k: &op::Kernel) -> Result<Self> {
        fn walk(ast: &Ast, arg_map: &HashMap<ArgId, ArgId>) -> Result<Ast> {
            use op::AstInner as A;
            match ast.inner.as_ref() {
                A::Id { .. } => crate::bail!("unexpected id node"),
                A::Load { src, layout } => {
                    let src = match arg_map.get(src) {
                        None => crate::bail!("BUG: missing arg id {src:?}"),
                        Some(id) => *id,
                    };
                    op::load(src, layout.clone(), ast.dtype)
                }
                A::Unary { op, arg } => {
                    let arg = walk(arg, arg_map)?;
                    op::unary(*op, arg)
                }
                A::Binary { op, lhs, rhs } => {
                    let lhs = walk(lhs, arg_map)?;
                    let rhs = walk(rhs, arg_map)?;
                    op::binary(*op, lhs, rhs)
                }
                A::Const(cst) => op::cst(*cst),
                A::Reduce { op, arg, axis } => {
                    let arg = walk(arg, arg_map)?;
                    op::reduce(*op, arg, *axis)
                }
                A::Broadcast { arg, broadcasted_dims: _ } => {
                    let arg = walk(arg, arg_map)?;
                    op::broadcast(arg, ast.shape())
                }
            }
        }

        let mut arg_map = HashMap::new();
        let mut args = Vec::with_capacity(k.args.len());
        let mut ops = Vec::with_capacity(k.ops.len());
        for (id, arg) in k.args.iter().enumerate() {
            let id = ArgId::from_usize(id);
            arg_map.insert(arg.id(), id);
            args.push(op::Arg::new(id, arg.type_()));
        }
        for op in k.ops.iter() {
            let op::Store { dst, layout, value } = op;
            let dst = match arg_map.get(dst) {
                None => crate::bail!("BUG: missing arg id {dst:?}"),
                Some(id) => *id,
            };
            let value = walk(value, &arg_map)?;
            ops.push(op::store(dst, layout.clone(), value)?)
        }
        Ok(Self { args, ops })
    }
}
