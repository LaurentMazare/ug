pub use crate::dtype::CpuStorage;
use crate::{DType, Result};
use half::{bf16, f16};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug)]
pub struct CpuDevice;

impl crate::Device for CpuDevice {
    type Slice = CpuStorage;
    type Func = Func;

    unsafe fn allocate_uninit<DT: crate::WithDType>(&self, len: usize) -> Result<Self::Slice> {
        let slice = match DT::DTYPE {
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; len]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; len]),
            DType::F32 => CpuStorage::F32(vec![0f32; len]),
            DType::I32 => CpuStorage::I32(vec![0i32; len]),
            DType::I64 => CpuStorage::I64(vec![0i64; len]),
        };
        Ok(slice)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn compile(&self, kernel: &crate::lang::ssa::Kernel) -> Result<Self::Func> {
        let mut c_code = Vec::with_capacity(8192);
        let pid = std::process::id();
        let kernel_id = KernelId::new().as_usize();
        let func_name = format!("ug_{pid}_{kernel_id}");
        crate::cpu_code_gen::gen(&mut c_code, &func_name, kernel)?;
        CpuDevice::compile(self, &c_code, func_name)
    }

    fn run(&self, f: &Self::Func, args: &mut [&mut Self::Slice]) -> Result<()> {
        use libloading::Symbol as S;
        use std::ffi::c_void;

        let func_name = f.func_name.as_bytes();
        match args {
            [] => {
                let symbol: S<unsafe extern "C" fn()> = unsafe { f.lib.get(func_name)? };
                unsafe { symbol() }
            }
            [arg1] => {
                let symbol: S<unsafe extern "C" fn(*mut c_void)> = unsafe { f.lib.get(func_name)? };
                unsafe { symbol(arg1.as_mut_ptr()) }
            }
            _ => crate::bail!("unsupported number of args for kernel {}", args.len()),
        }
        Ok(())
    }
}

impl crate::Slice for CpuStorage {
    type Device = CpuDevice;

    fn len(&self) -> usize {
        CpuStorage::len(self)
    }

    fn dtype(&self) -> crate::DType {
        CpuStorage::dtype(self)
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn copy_host_to_device<DT: crate::WithDType>(&mut self, src: &[DT]) -> Result<()> {
        use crate::dtype::CpuStorage as S;
        use crate::dtype::CpuStorageRef as C;
        let dtype = self.dtype();
        if src.len() != self.len() {
            crate::bail!("dtoh len mismatch, dst {}, len {}", self.len(), src.len())
        }
        match (self, DT::to_cpu_storage(src)) {
            (S::BF16(dst), C::BF16(src)) => dst.copy_from_slice(src),
            (S::F16(dst), C::F16(src)) => dst.copy_from_slice(src),
            (S::F32(dst), C::F32(src)) => dst.copy_from_slice(src),
            (S::I32(dst), C::I32(src)) => dst.copy_from_slice(src),
            (S::I64(dst), C::I64(src)) => dst.copy_from_slice(src),
            (_, _) => {
                crate::bail!("htod dtype mismatch, dst {dtype:?}, src {:?}", DT::DTYPE)
            }
        }
        Ok(())
    }

    fn copy_device_to_host<DT: crate::WithDType>(&self, dst: &mut [DT]) -> Result<()> {
        use crate::dtype::CpuStorage as S;
        use crate::dtype::CpuStorageRefMut as C;
        let dtype = self.dtype();
        if dst.len() != self.len() {
            crate::bail!("dtoh len mismatch, dst {}, len {}", dst.len(), self.len())
        }
        match (self, DT::to_cpu_storage_mut(dst)) {
            (S::BF16(src), C::BF16(dst)) => dst.copy_from_slice(src),
            (S::F16(src), C::F16(dst)) => dst.copy_from_slice(src),
            (S::F32(src), C::F32(dst)) => dst.copy_from_slice(src),
            (S::I32(src), C::I32(dst)) => dst.copy_from_slice(src),
            (S::I64(src), C::I64(dst)) => dst.copy_from_slice(src),
            (_, _) => crate::bail!("dtoh dtype mismatch, dst {:?}, src {dtype:?}", DT::DTYPE),
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(usize);

impl KernelId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

pub struct Func {
    func_name: String,
    lib: libloading::Library,
}

impl Func {
    pub fn name(&self) -> &str {
        self.func_name.as_str()
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn run0(&self) -> Result<()> {
        let func_name = self.func_name.as_bytes();
        let symbol: libloading::Symbol<unsafe extern "C" fn()> = self.lib.get(func_name)?;
        symbol();
        Ok(())
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn run3<T>(&self, v1: &mut [T], v2: &mut [T], v3: &mut [T]) -> Result<()> {
        use std::ffi::c_void;

        let func_name = self.func_name.as_bytes();
        let symbol: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void),
        > = self.lib.get(func_name)?;
        symbol(
            v1.as_mut_ptr() as *mut c_void,
            v2.as_mut_ptr() as *mut c_void,
            v3.as_mut_ptr() as *mut c_void,
        );
        Ok(())
    }
}

impl crate::CpuDevice {
    pub fn compile(&self, c_code: &[u8], func_name: String) -> Result<Func> {
        fn compile_inner(
            c_code: &[u8],
            func_name: String,
            tmp_c: &PathBuf,
            tmp_so: &PathBuf,
        ) -> Result<Func> {
            std::fs::write(tmp_c, c_code)?;
            // TODO: add some environment variable or other ways to set some flags.
            let output = std::process::Command::new("gcc")
                .arg(tmp_c)
                .args([
                    "-shared",
                    "-O3",
                    "-march=native",
                    "-ffast-math",
                    "-fomit-frame-pointer",
                    "-o",
                ])
                .arg(tmp_so)
                .output()?;

            if !output.status.success() {
                crate::bail!(
                    "compilation failed\nstdout:\n{}\nstderr:{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            }
            let lib = unsafe { libloading::Library::new(tmp_so)? };
            Ok(Func { func_name, lib })
        }

        let tmp_dir = std::env::temp_dir();
        let tmp_c = tmp_dir.join(format!("{func_name}.c"));
        let tmp_so = tmp_dir.join(format!("{func_name}.so"));
        let result = compile_inner(c_code, func_name, &tmp_c, &tmp_so);
        // Ensure that the temporary files are cleaned up, even on failures.
        if !crate::utils::KEEP_TMP.with(|b| *b) {
            let _ = std::fs::remove_file(tmp_c);
            let _ = std::fs::remove_file(tmp_so);
        }
        result
    }
}
