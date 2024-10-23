use anyhow::Result;
use ug::lang::ssa;

struct V(ssa::VarId);

impl std::fmt::Display for V {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__var{}", self.0.as_usize())
    }
}

struct C(ssa::Const);

impl std::fmt::Display for C {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::num::FpCategory;
        match self.0 {
            ssa::Const::I32(v) => write!(f, "{v}"),
            ssa::Const::F32(v) => match v.classify() {
                // Using the INFINITY / NAN macros would feel a bit better but they don't
                // seem available and I haven't find out how to include<cmath>.
                FpCategory::Nan => write!(f, "0. / 0."),
                FpCategory::Infinite if v > 0. => write!(f, "1. / 0."),
                FpCategory::Infinite => write!(f, "-1. / 0."),
                FpCategory::Zero | FpCategory::Normal | FpCategory::Subnormal => {
                    // We use the debug trait rather than display for floats as the outcome
                    // on f32::MIN would not round trip properly with display.
                    write!(f, "{v:?}")
                }
            },
        }
    }
}

struct A(ssa::A);

impl std::fmt::Display for A {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ssa::A::Var(v) => V(v).fmt(f),
            ssa::A::Const(c) => C(c).fmt(f),
        }
    }
}

struct D(ssa::DType);

impl std::fmt::Display for D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype = match self.0 {
            ssa::DType::F32 => "float",
            ssa::DType::I32 => "int",
            ssa::DType::PtrF32 => "float*",
            ssa::DType::PtrI32 => "int*",
        };
        f.write_str(dtype)
    }
}

pub fn gen<W: std::io::Write>(w: &mut W, func_name: &str, kernel: &ssa::Kernel) -> Result<()> {
    let contains_reduce_local =
        kernel.instrs.iter().any(|v| matches!(v, ssa::Instr::ReduceLocal { .. }));
    if contains_reduce_local {
        w.write_all(include_bytes!("reduce.cu"))?;
    }
    let mut args = std::collections::HashMap::new();
    for (idx, instr) in kernel.instrs.iter().enumerate() {
        if let ssa::Instr::DefineGlobal { index, dtype } = instr {
            args.insert(idx, (*index, *dtype));
        }
    }
    let mut args = args.into_iter().collect::<Vec<_>>();
    args.sort_by_key(|v| v.0);
    writeln!(w, "extern \"C\" __global__ void {func_name}(")?;
    for (arg_idx2, &(var_id, (arg_idx, dtype))) in args.iter().enumerate() {
        if arg_idx != arg_idx2 {
            anyhow::bail!("unexpected arguments in kernel {args:?}")
        }
        let is_last = arg_idx == args.len() - 1;
        let delim = if is_last { "" } else { "," };
        writeln!(w, "  {} {}{delim}", D(dtype), V(ssa::VarId::new(var_id)))?;
    }
    writeln!(w, ") {{")?;

    let mut depth = 0;
    for (var_id, instr) in kernel.instrs.iter().enumerate() {
        use ssa::Instr as I;
        let var_id = V(ssa::VarId::new(var_id));
        let indent = " ".repeat(2 * depth + 2);
        match instr {
            I::DefineGlobal { index: _, dtype: _ } => {}
            I::DefineLocal { dtype, size } => match dtype {
                // TODO(laurent): should we enforce the alignment in some cases?
                ssa::DType::PtrI32 => writeln!(w, "{indent}__shared__ int {var_id}[{size}];")?,
                ssa::DType::PtrF32 => writeln!(w, "{indent}__shared__ float {var_id}[{size}];")?,
                ssa::DType::F32 | ssa::DType::I32 => {
                    anyhow::bail!("unsupported dtype for DefineLocal {dtype:?}")
                }
            },
            I::DefineAcc(cst) | I::Const(cst) => match cst {
                ssa::Const::I32(v) => writeln!(w, "{indent}int {var_id} = {v};")?,
                ssa::Const::F32(v) => {
                    use std::num::FpCategory;
                    let v = match v.classify() {
                        // Using the INFINITY / NAN macros would feel a bit better but they don't
                        // seem available and I haven't find out how to include<cmath>.
                        FpCategory::Nan => "0. / 0.".to_string(),
                        FpCategory::Infinite if *v > 0. => "1. / 0.".to_string(),
                        FpCategory::Infinite => "-1. / 0.".to_string(),
                        FpCategory::Zero | FpCategory::Normal | FpCategory::Subnormal => {
                            // We use the debug trait rather than display for floats as the outcome
                            // on f32::MIN would not round trip properly with display.
                            format!("{v:?}")
                        }
                    };
                    writeln!(w, "{indent}float {var_id} = {v};")?
                }
            },
            I::If { cond, end_idx: _ } => {
                writeln!(w, "{indent}if ({}) {{", A(*cond),)?;
                depth += 1;
            }
            I::Range { lo, up, end_idx: _ } => {
                writeln!(
                    w,
                    "{indent}for (int {var_id} = {}; {var_id} < {}; ++{var_id}) {{",
                    A(*lo),
                    A(*up)
                )?;
                depth += 1;
            }
            I::EndIf | I::EndRange { start_idx: _ } => {
                if depth == 0 {
                    anyhow::bail!("unmatched EndRange")
                }
                depth -= 1;
                let indent = " ".repeat(2 * depth + 2);
                writeln!(w, "{indent}}}")?;
            }
            I::Load { src, offset, dtype } => {
                writeln!(w, "{indent}{} {var_id} = {}[{}];", D(*dtype), V(*src), A(*offset))?;
            }
            I::Assign { dst, src } => {
                writeln!(w, "{indent}{} = {};", V(*dst), A(*src))?;
            }
            I::Store { dst, offset, value, dtype: _ } => {
                writeln!(w, "{indent}{}[{}] = {};", V(*dst), A(*offset), A(*value))?;
            }
            I::Binary { op, lhs, rhs, dtype } => {
                let op = match op {
                    ssa::BinaryOp::Add => format!("{} + {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Mul => format!("{} * {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Sub => format!("{} - {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Div => format!("{} / {}", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Min => format!("min({}, {})", A(*lhs), A(*rhs)),
                    ssa::BinaryOp::Max => format!("max({}, {})", A(*lhs), A(*rhs)),
                };
                writeln!(w, "{indent}{} {var_id} = {op};", D(*dtype),)?;
            }
            I::Unary { op, arg, dtype } => {
                let op = match op {
                    ssa::UnaryOp::Exp => "__expf",
                    ssa::UnaryOp::Neg => "neg",
                    ssa::UnaryOp::Id => "",
                    ssa::UnaryOp::Cast => match dtype {
                        ssa::DType::F32 => "static_cast<float>",
                        ssa::DType::I32 => "static_cast<int>",
                        ssa::DType::PtrF32 => "static_cast<float*>",
                        ssa::DType::PtrI32 => "static_cast<int*>",
                    },
                };
                writeln!(w, "{indent}{} {var_id} = {op}({});", D(*dtype), A(*arg))?;
            }
            I::Special(ssa::Special::LocalIdx) => {
                writeln!(w, "{indent}int {var_id} = threadIdx.x;")?
            }
            I::Special(ssa::Special::GridIdx) => writeln!(w, "{indent}int {var_id} = blockIdx.x;")?,
            I::Barrier => writeln!(w, "{indent}__syncthreads();")?,
            I::ReduceLocal { op, arg, dtype } => {
                let op = match op {
                    ssa::ReduceOp::Sum => "block_reduce_sum",
                    ssa::ReduceOp::Min => "block_reduce_min",
                    ssa::ReduceOp::Max => "block_reduce_max",
                };
                writeln!(w, "{indent}{} {var_id} = {op}({});", D(*dtype), A(*arg))?;
            }
        }
    }
    writeln!(w, "}}")?;
    if depth > 0 {
        anyhow::bail!("unmatched Range")
    }
    Ok(())
}
