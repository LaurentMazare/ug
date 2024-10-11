use anyhow::Result;
use ug::lang::ssa;

struct Var(usize);

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__var{}", self.0)
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
        let dtype = match dtype {
            ssa::DType::F32 => "float",
            ssa::DType::I32 => "int",
            ssa::DType::PtrF32 => "float*",
            ssa::DType::PtrI32 => "int*",
        };
        writeln!(w, "  {dtype} {}{delim}", Var(var_id))?;
    }
    writeln!(w, ") {{")?;
    writeln!(w, "  int __pid0 = blockIdx.x * blockDim.x + threadIdx.x;")?;
    writeln!(w, "  int __pid1 = blockIdx.y * blockDim.y + threadIdx.y;")?;
    writeln!(w, "  int __pid2 = blockIdx.z * blockDim.z + threadIdx.z;")?;

    for (var_id, instr) in kernel.instrs.iter().enumerate() {
        use ssa::Instr as I;
        let var_id = Var(var_id);
        match instr {
            I::DefineGlobal { index: _, dtype: _ } => {}
            I::Const(cst) => match cst {
                ssa::Const::I32(v) => writeln!(w, "  int {var_id} = {v};")?,
                ssa::Const::F32(v) => writeln!(w, "  float {var_id} = {v};")?,
            },
            _ => anyhow::bail!("not implemented yet for cuda {instr:?}"),
        }
    }

    writeln!(w, "}}")?;
    Ok(())
}
