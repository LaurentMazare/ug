use crate::lang::op::Ast;
use crate::{Device, Layout, LazyBuffer, Result};

pub struct ScheduleItem<D: Device> {
    ast: Ast,
    dst: (crate::lang::ArgId, LazyBuffer<D>),
}

impl<D: Device> ScheduleItem<D> {
    pub fn into_ast(self) -> Ast {
        self.ast
    }

    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    pub fn kernel(&self) -> Result<crate::lang::op::Kernel> {
        use crate::lang::op;
        let ast = &self.ast;
        let dst = &self.dst;
        let arg = op::Arg::new(dst.0, crate::lang::Type::Ptr(dst.1.dtype()));
        let sto = op::store(arg.id(), dst.1.layout().clone(), ast.clone())?;
        // TODO: use the stored variables/args.
        let kernel = op::Kernel::new(format!("realize_{:?}", dst.1.id()), vec![arg], vec![sto]);
        Ok(kernel)
    }
}

pub struct Schedule<D: Device> {
    /// Elements in `items` are topologically sorted so that they can be run in order.
    items: Vec<ScheduleItem<D>>,
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
        for &buffer in buffers.iter() {
            let ast = context.walk(buffer)?;
            let dst_id = crate::lang::ArgId::new();
            context.items.push(ScheduleItem { ast, dst: (dst_id, buffer.clone()) });
        }
        Ok(Self { items: context.items, device })
    }

    pub fn create_one(buffer: &LazyBuffer<D>) -> Result<Self> {
        let mut context = Context::new();
        let ast = context.walk(buffer)?;
        let dst_id = crate::lang::ArgId::new();
        context.items.push(ScheduleItem { ast, dst: (dst_id, buffer.clone()) });
        Ok(Self { items: context.items, device: buffer.device().clone() })
    }

    pub fn items(&self) -> &[ScheduleItem<D>] {
        self.items.as_slice()
    }

    pub fn compile(&self) -> Result<CompiledSchedule<D>> {
        // TODO: compilation cache.
        let mut funcs = Vec::with_capacity(self.items().len());
        for item in self.items() {
            let kernel = item.kernel()?;
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

struct Context<D: Device> {
    items: Vec<ScheduleItem<D>>,
    // TODO: Detect the shared parts of the computation graphs and ensure that these are realized
    // and converted to kernel arguments.
}

impl<D: Device> Context<D> {
    fn new() -> Self {
        Self { items: vec![] }
    }

    fn walk(&mut self, b: &LazyBuffer<D>) -> Result<Ast> {
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
                let dst_id = crate::lang::ArgId::new();
                self.items.push(ScheduleItem { ast, dst: (dst_id, b.clone()) });
                crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
            }
        };

        Ok(ast)
    }
}
