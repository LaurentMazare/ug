use crate::lang::op::Ast;
use crate::{Device, Layout, LazyBuffer, Result};

#[derive(Debug)]
pub struct ScheduleItem {
    ast: Ast,
    // TODO: Add the buffers, probably in a lazily allocated way, see device.Buffer in tinygrad.
}

impl ScheduleItem {
    pub fn into_ast(self) -> Ast {
        self.ast
    }

    pub fn ast(&self) -> &Ast {
        &self.ast
    }
}

#[derive(Debug)]
pub struct Schedule<D: Device> {
    /// Elements in `items` are topologically sorted so that they can be run in order.
    items: Vec<ScheduleItem>,
    device: D,
    // TODO: Add variables.
}

impl<D: Device> Schedule<D> {
    pub fn create(buffers: &[&LazyBuffer<D>]) -> Result<Self> {
        let device = if buffers.is_empty() {
            crate::bail!("no buffers provided")
        } else {
            buffers[0].device().clone()
        };
        let mut context = Context::new();
        for buffer in buffers.iter() {
            let ast = context.walk(buffer)?;
            context.items.push(ScheduleItem { ast });
        }
        Ok(Self { items: context.items, device })
    }

    pub fn create_one(buffer: &LazyBuffer<D>) -> Result<Self> {
        let mut context = Context::new();
        let ast = context.walk(buffer)?;
        context.items.push(ScheduleItem { ast });
        Ok(Self { items: context.items, device: buffer.device().clone() })
    }

    pub fn items(&self) -> &[ScheduleItem] {
        self.items.as_slice()
    }

    pub fn compile(&self) -> Result<CompiledSchedule<D>> {
        use crate::lang::op;
        // TODO: compilation cache.
        let mut funcs = Vec::with_capacity(self.items().len());
        for item in self.items() {
            let ast = item.ast().clone();
            // TODO: use the stored variables/args.
            let arg = op::Arg::new(crate::lang::Type::Ptr(crate::DType::F32));
            let sto = op::store(arg.id(), Layout::from_shape((5, 2)), ast)?;
            let kernel = op::Kernel::new("forty_two".into(), vec![arg], vec![sto]);
            let ssa = kernel.lower(&Default::default())?;
            let func = self.device.compile(&ssa)?;
            funcs.push(func)
        }
        let device = self.device.clone();
        Ok(CompiledSchedule { funcs, device })
    }
}

pub struct CompiledSchedule<D: Device> {
    funcs: Vec<D::Func>,
    device: D,
}

impl<D: Device> CompiledSchedule<D> {
    pub fn run(&self) -> Result<()> {
        for func in self.funcs.iter() {
            // TODO: proper argument handling.
            self.device.run(func, &mut [])?
        }
        Ok(())
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
