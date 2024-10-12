use anyhow::Result;
use ug::lang::ssa;

struct V(usize);

impl std::fmt::Display for V {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__var{}", self.0)
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
        writeln!(w, "  {} {}{delim}", D(dtype), V(var_id))?;
    }
    writeln!(w, ") {{")?;

    let mut depth = 0;
    for (var_id, instr) in kernel.instrs.iter().enumerate() {
        use ssa::Instr as I;
        let var_id = V(var_id);
        let indent = " ".repeat(2 * depth + 2);
        match instr {
            I::DefineGlobal { index: _, dtype: _ } => {}
            I::DefineAcc(cst) | I::Const(cst) => match cst {
                ssa::Const::I32(v) => writeln!(w, "{indent}int {var_id} = {v};")?,
                ssa::Const::F32(v) => writeln!(w, "{indent}float {var_id} = {v};")?,
            },
            I::Range { lo, up, end_idx: _ } => {
                writeln!(
                    w,
                    "{indent}for (int {var_id} = {}; {var_id} < {}; ++{var_id}) {{",
                    V(lo.as_usize()),
                    V(up.as_usize())
                )?;
                depth += 1;
            }
            I::EndRange { start_idx: _ } => {
                if depth == 0 {
                    anyhow::bail!("unmatched EndRange")
                }
                depth -= 1;
                let indent = " ".repeat(2 * depth + 2);
                writeln!(w, "{indent}}}")?;
            }
            I::Load { src, offset, dtype } => {
                writeln!(
                    w,
                    "{indent}{} {var_id} = {}[{}];",
                    D(*dtype),
                    V(src.as_usize()),
                    V(offset.as_usize())
                )?;
            }
            I::Assign { dst, src } => {
                writeln!(w, "{indent}{} = {};", V(dst.as_usize()), V(src.as_usize()))?;
            }
            I::Store { dst, offset, value, dtype: _ } => {
                writeln!(
                    w,
                    "{indent}{}[{}] = {};",
                    V(dst.as_usize()),
                    V(offset.as_usize()),
                    V(value.as_usize())
                )?;
            }
            I::Binary { op, lhs, rhs, dtype } => {
                let op = match op {
                    ssa::BinaryOp::Add => "+",
                    ssa::BinaryOp::Mul => "*",
                    ssa::BinaryOp::Sub => "-",
                    ssa::BinaryOp::Div => "/",
                };
                writeln!(
                    w,
                    "{indent}{} {var_id} = {} {op} {};",
                    D(*dtype),
                    V(lhs.as_usize()),
                    V(rhs.as_usize())
                )?;
            }
            I::Unary { op, arg, dtype } => {
                let op = match op {
                    ssa::UnaryOp::Exp => "exp",
                    ssa::UnaryOp::Neg => "neg",
                };
                writeln!(w, "{indent}{} {var_id} = {op}({});", D(*dtype), V(arg.as_usize()),)?;
            }
            I::Special(ssa::Special::LocalIdx) => {
                writeln!(w, "{indent}int {var_id} = threadIdx.x;")?
            }
            I::Special(ssa::Special::GridIdx) => writeln!(w, "{indent}int {var_id} = blockIdx.x;")?,
            I::Barrier => writeln!(w, "{indent}__synchthreads();")?,
        }
    }
    writeln!(w, "}}")?;
    if depth > 0 {
        anyhow::bail!("unmatched Range")
    }
    Ok(())
}
