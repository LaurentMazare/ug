use crate::lang::op::Ast;
use crate::{Device, Layout, LazyBuffer, Result};

#[derive(Debug)]
pub struct ScheduleItem {
    pub ast: Ast,
    // TODO: Add the buffers, probably in a lazily allocated way, see device.Buffer in tinygrad.
}

#[derive(Debug)]
pub struct Schedule {
    /// Elements in `items` are topologically sorted so that they can be run in order.
    pub items: Vec<ScheduleItem>,
    // TODO: Add variables.
}

impl Schedule {
    pub fn create<D: Device>(buffers: &[&LazyBuffer<D>]) -> Result<Schedule> {
        let mut context = Context::new();
        for buffer in buffers.iter() {
            let ast = context.walk(buffer)?;
            context.items.push(ScheduleItem { ast });
        }
        Ok(Self { items: context.items })
    }

    pub fn create_one<D: Device>(buffer: &LazyBuffer<D>) -> Result<Schedule> {
        let mut context = Context::new();
        let ast = context.walk(buffer)?;
        context.items.push(ScheduleItem { ast });
        Ok(Self { items: context.items })
    }
}

struct Context {
    items: Vec<ScheduleItem>,
    // TODO: Detect the shared parts of the computation graphs and ensure that these are realized
    // and converted to kernel arguments.
}

impl Context {
    fn new() -> Self {
        Self { items: vec![] }
    }

    fn walk<D: Device>(&mut self, b: &LazyBuffer<D>) -> Result<Ast> {
        use crate::lazy_buffer::Op;

        let dtype = b.dtype();
        let shape = b.shape();
        let ast = match b.op() {
            Op::Unary(op, arg) => {
                let ast = self.walk(arg)?;
                crate::lang::op::unary(*op, ast)?
            }
            Op::Binary(op, lhs, rhs) => {
                let lhs = self.walk(lhs)?;
                let rhs = self.walk(rhs)?;
                crate::lang::op::binary(*op, lhs, rhs)?
            }
            Op::Reduce(op, arg, axis) => {
                let ast = self.walk(arg)?;
                crate::lang::op::reduce(*op, ast, *axis)?
            }
            Op::Const(cst) => crate::lang::op::cst(*cst),
            Op::Copy(_sto) => {
                let arg_id = crate::lang::ArgId::new();
                crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
            }
            Op::Layout(_op, arg) => {
                let ast = self.walk(arg)?;
                self.items.push(ScheduleItem { ast });
                let arg_id = crate::lang::ArgId::new();
                crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
            }
        };

        Ok(ast)
    }
}
