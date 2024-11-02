use crate::lang::op::{ArgId, Ast};
use crate::{Device, Layout, LazyBuffer, Result};

type Args<D> = Vec<(ArgId, LazyBuffer<D>)>;

pub struct KernelItem<D: Device> {
    ast: Ast,
    dst: (ArgId, LazyBuffer<D>),
    args: Args<D>,
}

impl<D: Device> KernelItem<D> {
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
        let args = self
            .args
            .iter()
            .map(|(id, lb)| op::Arg::new(*id, crate::lang::Type::Ptr(lb.dtype())))
            .collect::<Vec<_>>();
        let sto = op::store(dst.0, dst.1.layout().clone(), ast.clone())?;
        let kernel = op::Kernel::new(format!("realize_{:?}", dst.1.id()), args, vec![sto]);
        Ok(kernel)
    }
}

pub enum ScheduleItem<D: Device> {
    Kernel(KernelItem<D>),
    MatMul { dst: LazyBuffer<D>, lhs: LazyBuffer<D>, rhs: LazyBuffer<D> },
}

pub struct Schedule<D: Device> {
    /// Elements in `items` are topologically sorted so that they can be run in order.
    items: Vec<ScheduleItem<D>>,
    per_arg_id: std::collections::HashMap<ArgId, LazyBuffer<D>>,
    device: D,
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
            context.push_schedule_item(buffer)?;
        }
        Ok(Self { items: context.items, device, per_arg_id: context.per_arg_id })
    }

    pub fn create_one(buffer: &LazyBuffer<D>) -> Result<Self> {
        let mut context = Context::new();
        context.push_schedule_item(buffer)?;
        Ok(Self {
            items: context.items,
            device: buffer.device().clone(),
            per_arg_id: context.per_arg_id,
        })
    }

    pub fn items(&self) -> &[ScheduleItem<D>] {
        self.items.as_slice()
    }

    pub fn compile(&self) -> Result<CompiledSchedule<D>> {
        // TODO: compilation cache.
        let mut funcs = Vec::with_capacity(self.items().len());
        for item in self.items() {
            let call = match item {
                ScheduleItem::MatMul { dst, lhs, rhs } => {
                    Func::MatMul { dst: dst.clone(), lhs: lhs.clone(), rhs: rhs.clone() }
                }
                ScheduleItem::Kernel(item) => {
                    let kernel = item.kernel()?;
                    let ssa = kernel.lower(&Default::default())?;
                    let mut args = vec![];
                    for arg in ssa.args().iter() {
                        let arg_id = arg.0.id();
                        let arg = match self.per_arg_id.get(&arg_id) {
                            Some(b) => b.clone(),
                            None => crate::bail!("no arg for id {arg_id:?}"),
                        };
                        args.push((arg_id, arg))
                    }
                    let func = self.device.compile(&ssa)?;
                    Func::Kernel { func, args }
                }
            };
            funcs.push(call)
        }
        let device = self.device.clone();
        Ok(CompiledSchedule { funcs, device })
    }
}

pub enum Func<D: Device> {
    Kernel { func: D::Func, args: Args<D> },
    MatMul { dst: LazyBuffer<D>, lhs: LazyBuffer<D>, rhs: LazyBuffer<D> },
}

pub struct CompiledSchedule<D: Device> {
    funcs: Vec<Func<D>>,
    device: D,
}

impl<D: Device> CompiledSchedule<D> {
    pub fn run(&self) -> Result<()> {
        for func in self.funcs.iter() {
            match func {
                Func::Kernel { func, args } => {
                    // Should we do some deadlock detection?
                    let mut locks = args
                        .iter()
                        .map(|(_id, lb)| {
                            unsafe { lb.maybe_allocate_uninit()? };
                            let lock = lb.data().lock()?;
                            Ok(lock)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut locks =
                        locks.iter_mut().map(|v| v.as_mut().unwrap()).collect::<Vec<_>>();
                    self.device.run(func, &mut locks)?
                }
                Func::MatMul { dst: _, lhs: _, rhs: _ } => {
                    todo!()
                }
            }
        }
        Ok(())
    }
}

struct Context<D: Device> {
    items: Vec<ScheduleItem<D>>,
    per_arg_id: std::collections::HashMap<ArgId, LazyBuffer<D>>,
    // TODO: Detect the shared parts of the computation graphs and ensure that these are realized
    // and converted to kernel arguments.
}

impl<D: Device> Context<D> {
    fn new() -> Self {
        Self { items: vec![], per_arg_id: std::collections::HashMap::new() }
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
            Op::MatMul(lhs, rhs) => {
                let _lhs = self.walk(lhs)?;
                let _rhs = self.walk(rhs)?;
                // TODO: Split the graph here.
                todo!()
            }
            Op::Reduce(op, arg, axis) => {
                let ast = self.walk(arg)?;
                crate::lang::op::reduce(*op, ast, *axis)?
            }
            Op::Const(cst) => crate::lang::op::cst(*cst),
            Op::Copy(_sto) => {
                let arg_id = ArgId::new();
                self.per_arg_id.insert(arg_id, b.clone());
                // TODO: Add to args, and handle const properly.
                crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
            }
            Op::Layout(_op, arg) => {
                let dst_id = self.push_schedule_item(arg)?;
                crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
            }
        };
        Ok(ast)
    }

    fn push_schedule_item(&mut self, buffer: &LazyBuffer<D>) -> Result<ArgId> {
        let ast = self.walk(buffer)?;
        let dst_id = ArgId::new();
        self.per_arg_id.insert(dst_id, buffer.clone());
        let mut arg_ids = ast.arg_ids();
        arg_ids.insert(dst_id);
        let args = arg_ids
            .into_iter()
            .map(|arg_id| {
                let arg = match self.per_arg_id.get(&arg_id) {
                    Some(b) => b.clone(),
                    None => crate::bail!("no arg for id {arg_id:?}"),
                };
                Ok((arg_id, arg))
            })
            .collect::<Result<Vec<_>>>()?;
        let si = KernelItem { ast, dst: (dst_id, buffer.clone()), args };
        self.items.push(ScheduleItem::Kernel(si));
        Ok(dst_id)
    }
}
