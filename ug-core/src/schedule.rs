use crate::lang::op::{ArgId, Ast};
use crate::{Device, Layout, LazyBuffer, Result};
use std::collections::HashMap;

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
        let dst_layout = Layout::from_shape(dst.1.shape());
        let sto = op::store(dst.0, dst_layout, ast.clone())?;
        let kernel = op::Kernel::new(format!("realize_{:?}", dst.1.id()), args, vec![sto]);
        Ok(kernel)
    }
}

pub enum ScheduleItem<D: Device> {
    Kernel(KernelItem<D>),
    MatMul {
        dst: LazyBuffer<D>,
        lhs: LazyBuffer<D>,
        rhs: LazyBuffer<D>,
        bmnk: (usize, usize, usize, usize),
        transpose: bool,
    },
    Custom {
        f: crate::lazy_buffer::CustomF<D::Slice>,
        args: Args<D>,
    },
}

pub struct Schedule<D: Device> {
    /// Elements in `items` are topologically sorted so that they can be run in order.
    items: Vec<ScheduleItem<D>>,
    per_arg_id: HashMap<ArgId, LazyBuffer<D>>,
    span_compile: tracing::Span,
    span_kernel: tracing::Span,
    device: D,
}

impl<D: Device> Schedule<D> {
    pub fn get_arg_id(&self, arg_id: ArgId) -> Result<&LazyBuffer<D>> {
        match self.per_arg_id.get(&arg_id) {
            Some(b) => Ok(b),
            None => crate::bail!("no arg for id {arg_id:?}"),
        }
    }

    pub fn create(buffers: &[&LazyBuffer<D>]) -> Result<Self> {
        let device = if buffers.is_empty() {
            crate::bail!("no buffers provided")
        } else {
            buffers[0].device().clone()
        };
        let mut cnts = HashMap::new();
        for buffer in buffers.iter() {
            id_cnts(buffer, &mut cnts)
        }
        let mut context = Context::new(cnts);
        for &buffer in buffers.iter() {
            context.push_schedule_item(buffer)?;
        }
        let span_compile = tracing::span!(tracing::Level::TRACE, "compile");
        let span_kernel = tracing::span!(tracing::Level::TRACE, "kernel");
        Ok(Self {
            items: context.items,
            device,
            per_arg_id: context.per_arg_id,
            span_compile,
            span_kernel,
        })
    }

    pub fn create_one(buffer: &LazyBuffer<D>) -> Result<Self> {
        Self::create(&[buffer])
    }

    pub fn items(&self) -> &[ScheduleItem<D>] {
        self.items.as_slice()
    }

    pub fn compile(&self) -> Result<CompiledSchedule<D>> {
        self.compile_with_cache(&mut Default::default())
    }

    pub fn compile_with_cache(
        &self,
        compilation_cache: &mut crate::cache::CompilationCache<D>,
    ) -> Result<CompiledSchedule<D>> {
        let _guard = self.span_compile.enter();
        let mut funcs = Vec::with_capacity(self.items().len());
        for item in self.items() {
            let call = match item {
                ScheduleItem::MatMul { dst, lhs, rhs, bmnk, transpose } => Func::MatMul {
                    dst: dst.clone(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    bmnk: *bmnk,
                    transpose: *transpose,
                },
                ScheduleItem::Custom { f, args } => {
                    Func::Custom { f: f.clone(), args: args.to_vec() }
                }
                ScheduleItem::Kernel(item) => {
                    let kernel = item.kernel()?;
                    let norm_kernel = crate::cache::NormalizedKernel::new(&kernel)?;
                    if let Some(func) = compilation_cache.get(&norm_kernel) {
                        // TODO: Simplify args handling so as to ensure that this matches the
                        // construction of args using ssa.args() below.
                        let mut args = vec![];
                        for arg in kernel.args.iter() {
                            let arg_id = arg.id();
                            let arg = self.get_arg_id(arg_id)?;
                            args.push((arg_id, arg.clone()))
                        }
                        Func::Kernel { func, args }
                    } else {
                        let _guard = self.span_kernel.enter();
                        let ssa = kernel.lower(&Default::default())?;
                        let mut args = vec![];
                        for arg in ssa.args().iter() {
                            let arg_id = arg.0.id();
                            let arg = self.get_arg_id(arg_id)?;
                            args.push((arg_id, arg.clone()))
                        }
                        let func = self.device.compile(&ssa)?;
                        let func = std::sync::Arc::new(func);
                        compilation_cache.insert(norm_kernel, func.clone());
                        Func::Kernel { func, args }
                    }
                }
            };
            funcs.push(call)
        }
        let device = self.device.clone();
        Ok(CompiledSchedule { funcs, device })
    }
}

pub enum Func<D: Device> {
    Kernel {
        func: std::sync::Arc<D::Func>,
        args: Args<D>,
    },
    MatMul {
        dst: LazyBuffer<D>,
        lhs: LazyBuffer<D>,
        rhs: LazyBuffer<D>,
        bmnk: (usize, usize, usize, usize),
        transpose: bool,
    },
    Custom {
        f: crate::lazy_buffer::CustomF<D::Slice>,
        args: Args<D>,
    },
}

pub struct CompiledSchedule<D: Device> {
    funcs: Vec<Func<D>>,
    device: D,
}

impl<D: Device> CompiledSchedule<D> {
    pub fn run(&self) -> Result<()> {
        let span_mm = tracing::span!(tracing::Level::TRACE, "mm");
        let span_k = tracing::span!(tracing::Level::TRACE, "kernel");
        let span_custom = tracing::span!(tracing::Level::TRACE, "custom");
        // TODO: We should avoid re-running kernels that have unchanged inputs, tracking
        // variables/copies is likely enough for this.
        for func in self.funcs.iter() {
            match func {
                Func::Kernel { func, args } => {
                    let _guard = span_k.enter();
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
                Func::MatMul { dst, lhs, rhs, bmnk, transpose } => {
                    let _guard = span_mm.enter();
                    let lhs_dims = lhs.dims();
                    let lhs_rank = lhs.rank();
                    let rhs_dims = rhs.dims();
                    let rhs_rank = rhs.rank();

                    let lhs_l = if lhs_rank < rhs_rank {
                        let lhs_dims = [&vec![1; rhs_rank - lhs_rank], lhs_dims].concat();
                        crate::Layout::from_shape(lhs_dims)
                    } else {
                        crate::Layout::from_shape(lhs_dims)
                    };
                    let rhs_l = if rhs_rank < lhs_rank {
                        let rhs_dims = [&vec![1; lhs_rank - rhs_rank], rhs_dims].concat();
                        crate::Layout::from_shape(rhs_dims)
                    } else {
                        crate::Layout::from_shape(rhs_dims)
                    };
                    let rhs_l = if *transpose { rhs_l.transpose() } else { rhs_l };
                    // TODO: provide a nicer api on LazyBuffer to get the underlying buffer and
                    // have it created if necessary.
                    unsafe { dst.maybe_allocate_uninit()? };
                    unsafe { lhs.maybe_allocate_uninit()? };
                    unsafe { rhs.maybe_allocate_uninit()? };
                    let mut dst = dst.data().lock()?;
                    let dst = dst.as_mut().unwrap();
                    let lhs = lhs.data().lock()?;
                    let lhs = lhs.as_ref().unwrap();
                    let rhs = rhs.data().lock()?;
                    let rhs = rhs.as_ref().unwrap();
                    self.device.matmul(dst, lhs, rhs, *bmnk, &lhs_l, &rhs_l)?;
                }
                Func::Custom { f, args } => {
                    let _guard = span_custom.enter();
                    let mut locks = args
                        .iter()
                        .map(|(_id, lb)| {
                            unsafe { lb.maybe_allocate_uninit()? };
                            let lock = lb.data().lock()?;
                            Ok(lock)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let locks = locks.iter_mut().map(|v| v.as_mut().unwrap()).collect::<Vec<_>>();
                    f(locks)?
                }
            }
        }
        Ok(())
    }
}

struct Context<D: Device> {
    items: Vec<ScheduleItem<D>>,
    per_arg_id: HashMap<ArgId, LazyBuffer<D>>,
    ast_cache: HashMap<crate::lazy_buffer::Id, Ast>,
    id_cnts: HashMap<crate::lazy_buffer::Id, usize>,
}

impl<D: Device> Context<D> {
    fn new(id_cnts: HashMap<crate::lazy_buffer::Id, usize>) -> Self {
        Self { items: vec![], per_arg_id: HashMap::new(), ast_cache: HashMap::new(), id_cnts }
    }

    fn get_arg_id(&self, arg_id: ArgId) -> Result<&LazyBuffer<D>> {
        match self.per_arg_id.get(&arg_id) {
            Some(b) => Ok(b),
            None => crate::bail!("no arg for id {arg_id:?}"),
        }
    }

    fn walk(&mut self, b: &LazyBuffer<D>) -> Result<Ast> {
        use crate::lazy_buffer::Op;

        let id = b.id();
        if let Some(ast) = self.ast_cache.get(&id) {
            return Ok(ast.clone());
        }

        let dtype = b.dtype();
        let shape = b.shape();
        let ast = if b.realized() {
            let arg_id = ArgId::new();
            self.per_arg_id.insert(arg_id, b.clone());
            crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
        } else {
            match b.op() {
                Op::Unary(op, arg) => {
                    let ast = self.walk(arg)?;
                    crate::lang::op::unary(*op, ast)?
                }
                Op::Binary(op, lhs, rhs) => {
                    let lhs = self.walk(lhs)?;
                    let rhs = self.walk(rhs)?;
                    crate::lang::op::binary(*op, lhs, rhs)?
                }
                Op::MatMul(lhs, rhs, bmnk, transpose) => {
                    // MatMul currently aren't fused with the rest of the graph. Maybe we should
                    // allow for custom ops that would be handled in the same way.
                    let _lhs_id = self.push_schedule_item(lhs)?;
                    let _rhs_id = self.push_schedule_item(rhs)?;
                    let dst_id = ArgId::new();
                    self.per_arg_id.insert(dst_id, b.clone());
                    self.items.push(ScheduleItem::MatMul {
                        dst: b.clone(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        bmnk: *bmnk,
                        transpose: *transpose,
                    });
                    crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
                }
                Op::Reduce(op, arg, axis) => {
                    let ast = self.walk(arg)?;
                    crate::lang::op::reduce(*op, ast, *axis)?
                }
                Op::Const(cst) => crate::lang::op::cst(*cst)?,
                Op::Copy => {
                    let arg_id = ArgId::new();
                    self.per_arg_id.insert(arg_id, b.clone());
                    crate::lang::op::load(arg_id, Layout::from_shape(shape), dtype)?
                }
                Op::Layout(op, arg) => {
                    use crate::lazy_buffer::LayoutOp as L;
                    match op {
                        L::Noop | L::Reshape => {
                            let dst_id = self.push_schedule_item(arg)?;
                            crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
                        }
                        L::Broadcast => {
                            let ast = self.walk(arg)?;
                            crate::lang::op::broadcast(ast, b.shape())?
                        }
                    }
                }
                Op::Custom { f, args: b_args } => {
                    let mut args = Vec::with_capacity(b_args.len() + 1);
                    for arg in b_args.iter() {
                        let arg_id = self.push_schedule_item(arg)?;
                        args.push((arg_id, arg.clone()))
                    }
                    let dst_id = ArgId::new();
                    self.per_arg_id.insert(dst_id, b.clone());
                    args.push((dst_id, b.clone()));
                    self.items.push(ScheduleItem::Custom { f: f.clone(), args });
                    crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
                }
            }
        };
        // When a subtree appears multiple times in the ast, generate a dedicated kernel.
        let ast = if self.id_cnts.get(&id).copied().unwrap_or(0) > 1 {
            let dst_id = self.push_kernel(b, ast)?;
            crate::lang::op::load(dst_id, Layout::from_shape(shape), dtype)?
        } else {
            ast
        };
        self.ast_cache.insert(id, ast.clone());
        Ok(ast)
    }

    fn push_kernel(&mut self, buffer: &LazyBuffer<D>, ast: Ast) -> Result<ArgId> {
        if let crate::lang::op::AstInner::Load { src: src_arg_id, .. } = ast.inner.as_ref() {
            let src = self.get_arg_id(*src_arg_id)?;
            if src.id() == buffer.id() {
                // Avoid the cases where we load and store immediately a buffer, this is a no-op
                // and would result in a deadlock.
                return Ok(*src_arg_id);
            }
        }

        let dst_id = ArgId::new();
        self.per_arg_id.insert(dst_id, buffer.clone());
        let mut arg_ids = ast.arg_ids();
        arg_ids.insert(dst_id);
        let args = arg_ids
            .into_iter()
            .map(|arg_id| {
                let arg = self.get_arg_id(arg_id)?;
                Ok((arg_id, arg.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let si = KernelItem { ast, dst: (dst_id, buffer.clone()), args };
        self.items.push(ScheduleItem::Kernel(si));
        Ok(dst_id)
    }

    fn push_schedule_item(&mut self, buffer: &LazyBuffer<D>) -> Result<ArgId> {
        let ast = self.walk(buffer)?;
        self.push_kernel(buffer, ast)
    }
}

/// Return the number of uses for each buffer that is reachable from b. The number of uses can be
/// either 1 or 2 for the case where the buffer is used twice or more.
/// Note that realized nodes stop the propagation.
fn id_cnts<D: Device>(b: &LazyBuffer<D>, cnts: &mut HashMap<crate::lazy_buffer::Id, usize>) {
    use crate::lazy_buffer::Op;

    if b.realized() {
        return;
    }

    let id = b.id();
    let cnt = cnts.entry(id).or_insert(0);
    *cnt += 1;
    if *cnt > 1 {
        return;
    }
    match b.op() {
        Op::Copy | Op::Const(_) => {}
        Op::Layout(_, arg) | Op::Reduce(_, arg, _) | Op::Unary(_, arg) => id_cnts(arg, cnts),
        Op::MatMul(arg1, arg2, _, _) | Op::Binary(_, arg1, arg2) => {
            id_cnts(arg1, cnts);
            id_cnts(arg2, cnts);
        }
        Op::Custom { f: _, args } => {
            for arg in args.iter() {
                id_cnts(arg, cnts);
            }
        }
    }
}
