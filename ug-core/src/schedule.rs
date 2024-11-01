use crate::lang::op::{ArgId, Ast};
use crate::{Device, Layout, LazyBuffer, Result};

type Args<D> = Vec<(ArgId, LazyBuffer<D>)>;

pub struct ScheduleItem<D: Device> {
    ast: Ast,
    dst: (ArgId, LazyBuffer<D>),
    args: Args<D>,
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
        let mut args = self
            .args
            .iter()
            .map(|(id, lb)| op::Arg::new(*id, crate::lang::Type::Ptr(lb.dtype())))
            .collect::<Vec<_>>();
        args.push(op::Arg::new(dst.0, crate::lang::Type::Ptr(dst.1.dtype())));
        let sto = op::store(dst.0, dst.1.layout().clone(), ast.clone())?;
        // TODO: use the stored variables/args.
        let kernel = op::Kernel::new(format!("realize_{:?}", dst.1.id()), args, vec![sto]);
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
            let dst_id = ArgId::new();
            let args = context.take_args();
            context.items.push(ScheduleItem { ast, dst: (dst_id, buffer.clone()), args });
        }
        Ok(Self { items: context.items, device })
    }

    pub fn create_one(buffer: &LazyBuffer<D>) -> Result<Self> {
        let mut context = Context::new();
        let ast = context.walk(buffer)?;
        let dst_id = ArgId::new();
        let args = context.take_args();
        context.items.push(ScheduleItem { ast, dst: (dst_id, buffer.clone()), args });
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
            let args = vec![item.dst.clone()];
            funcs.push((func, args))
        }
        let device = self.device.clone();
        Ok(CompiledSchedule { funcs, device })
    }
}

pub struct CompiledSchedule<D: Device> {
    funcs: Vec<(D::Func, Args<D>)>,
    device: D,
}

impl<D: Device> CompiledSchedule<D> {
    pub fn run(&self) -> Result<()> {
        for (func, args) in self.funcs.iter() {
            // Should we do some deadlock detection?
            let mut locks = args
                .iter()
                .map(|(_id, lb)| {
                    unsafe { lb.maybe_allocate_uninit()? };
                    let lock = lb.data().lock()?;
                    Ok(lock)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut locks = locks.iter_mut().map(|v| v.as_mut().unwrap()).collect::<Vec<_>>();
            self.device.run(func, &mut locks)?
        }
        Ok(())
    }
}

struct Context<D: Device> {
    items: Vec<ScheduleItem<D>>,
    args: Args<D>,
    // TODO: Detect the shared parts of the computation graphs and ensure that these are realized
    // and converted to kernel arguments.
}

impl<D: Device> Context<D> {
    fn new() -> Self {
        Self { items: vec![], args: vec![] }
    }

    fn take_args(&mut self) -> Args<D> {
        let mut args = vec![];
        std::mem::swap(&mut args, &mut self.args);
        args
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
                let arg_id = ArgId::new();
                // TODO: Add to args, and handle const properly.
                crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
            }
            Op::Layout(_op, arg) => {
                let ast = self.walk(arg)?;
                let dst_id = ArgId::new();
                let args = self.take_args();
                self.items.push(ScheduleItem { ast, dst: (dst_id, b.clone()), args });
                self.args.push((dst_id, b.clone()));
                crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
            }
        };
        Ok(ast)
    }
}
