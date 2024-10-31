#![allow(unused)]
use crate::Result;
use std::path::PathBuf;

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

    pub unsafe fn run0(&self) -> Result<()> {
        let func_name = self.func_name.as_bytes();
        let symbol: libloading::Symbol<unsafe extern "C" fn()> =
            unsafe { self.lib.get(func_name)? };
        unsafe { symbol() };
        Ok(())
    }

    pub fn run3(&self, v1: &mut [i32], v2: &mut [i32], v3: &mut [i32]) -> Result<()> {
        let func_name = self.func_name.as_bytes();
        let symbol: libloading::Symbol<unsafe extern "C" fn(*mut i32, *mut i32, *mut i32)> =
            unsafe { self.lib.get(func_name)? };
        unsafe { symbol(v1.as_mut_ptr(), v2.as_mut_ptr(), v3.as_mut_ptr()) };
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
            let output = std::process::Command::new("gcc")
                .arg(tmp_c)
                .args(["-shared", "-O3", "-march=native", "-flto", "-fomit-frame-pointer", "-o"])
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
        let pid = std::process::id();
        let kernel_id = KernelId::new().as_usize();
        let tmp_c = tmp_dir.join(format!("ug_{pid}_{kernel_id}.c"));
        let tmp_so = tmp_dir.join(format!("ug_{pid}_{kernel_id}.so"));
        let result = compile_inner(c_code, func_name, &tmp_c, &tmp_so);
        // Ensure that the temporary files are cleaned up, even on failures.
        let _ = std::fs::remove_file(tmp_c);
        let _ = std::fs::remove_file(tmp_so);
        result
    }
}
