# TEAM-510: ug-rocm Implementation

**Date:** 2025-11-13  
**Status:** 🚧 IN PROGRESS  
**Goal:** Create ug-rocm backend to enable ROCm custom operations in Candle

## What We're Doing

Creating the missing `ug-rocm` crate to enable user-defined GPU kernels for ROCm in Candle. This is the missing piece that blocks `UgIOp1` from working with ROCm.

## Repository Setup

✅ **Cloned fork:** `git@github.com:veighnsche/ug.git` (branch: `rocm-support`)  
✅ **Added as submodule:** `/home/vince/Projects/rbee/deps/ug`  
✅ **Created ug-rocm crate:** `/home/vince/Projects/rbee/deps/ug/ug-rocm/`

## Files Created/Modified

### 1. ✅ Cargo.toml
**Location:** `/home/vince/Projects/rbee/deps/ug/ug-rocm/Cargo.toml`

```toml
[package]
name = "ug-rocm"
version.workspace = true
edition.workspace = true

[dependencies]
rocm-rs = { path = "../../rocm-rs" }
half = { workspace = true }
serde = { workspace = true }
thiserror = { workspace = true }
ug = { workspace = true }
```

**Key Decision:** Use `rocm-rs` instead of `cudarc` for ROCm bindings.

### 2. ✅ lib.rs
**Location:** `/home/vince/Projects/rbee/deps/ug/ug-rocm/src/lib.rs`

```rust
// Created by: TEAM-510 | ROCm backend for ug kernel compiler
pub use rocm_rs;
pub mod code_gen;
pub mod runtime;

pub use runtime::{Device, DeviceSlice};
```

**Changes:**
- Removed `gemm` module (not needed initially)
- Export `rocm_rs` instead of `cudarc`

### 3. ✅ code_gen.rs (Partially Updated)
**Location:** `/home/vince/Projects/rbee/deps/ug/ug-rocm/src/code_gen.rs`

**Key Changes:**
- Updated `DType` formatting for HIP types:
  - `BF16` → `hip_bfloat16` (instead of `__nv_bfloat16`)
  - `F16` → `_Float16` (instead of `__half`)
  - Rest remain the same (float, int, long long)

**What Still Needs Work:**
- The rest of the file is identical to CUDA version
- HIP is very similar to CUDA, so most code should work as-is
- May need minor adjustments for HIP-specific intrinsics

### 4. ✅ runtime.rs (Partially Updated)
**Location:** `/home/vince/Projects/rbee/deps/ug/ug-rocm/src/runtime.rs`

**Key Changes:**
```rust
// Updated imports
pub use rocm_rs::hip::{
    Function as HipFunction, 
    Stream as HipStream, 
    DeviceMemory,
};
pub use rocm_rs::hip::utils::Dim3;

// Updated error handling
impl<T> WithErr for std::result::Result<T, rocm_rs::hip::error::Error>
impl<T> WithErr for std::result::Result<T, rocm_rs::rocblas::error::Error>

// Updated Func struct
pub struct Func {
    func: HipFunction,
    stream: Arc<HipStream>,
    grid_dim: Dim3,
    block_dim: Dim3,
}

// Updated Device struct
pub struct Device {
    device: Arc<rocm_rs::hip::Device>,
    stream: Arc<rocm_rs::hip::Stream>,
    blas: Arc<rocm_rs::rocblas::RocBlas>,
}

// Updated SliceInner
pub enum SliceInner {
    F32(rocm_rs::hip::DeviceMemory<f32>),
    F16(rocm_rs::hip::DeviceMemory<half::f16>),
    BF16(rocm_rs::hip::DeviceMemory<half::bf16>),
    I32(rocm_rs::hip::DeviceMemory<i32>),
    I64(rocm_rs::hip::DeviceMemory<i64>),
}
```

**What Still Needs Work:**
- Rest of the file (lines 115-405) still references `cudarc`
- Need to implement `Device` methods
- Need to implement `Slice` methods
- Need to implement `ToSlice` trait

## Architecture Comparison

### CUDA (ug-cuda)
```
ug::lang::ssa::Kernel
    ↓ (ug_cuda::code_gen::gen)
CUDA C++ source code
    ↓ (cudarc::nvrtc::compile_ptx)
PTX bytecode
    ↓ (CudaContext::load_module)
CudaFunction
```

### ROCm (ug-rocm) - What We're Building
```
ug::lang::ssa::Kernel
    ↓ (ug_rocm::code_gen::gen)
HIP C++ source code
    ↓ (rocm_rs::hip::module::compile_and_load)
HSACO bytecode
    ↓ (Module::get_function)
HipFunction
```

## Key Differences: CUDA vs HIP

| Aspect | CUDA | HIP |
|--------|------|-----|
| Compiler | nvrtc (PTX) | hipcc (HSACO) |
| BF16 type | `__nv_bfloat16` | `hip_bfloat16` |
| F16 type | `__half` | `_Float16` |
| Function type | `CudaFunction` | `hipFunction_t` |
| Stream type | `CudaStream` | `hipStream_t` |
| Memory type | `CudaSlice<T>` | `DeviceMemory<T>` |
| Launch config | `LaunchConfig` | `Dim3` (grid + block) |

## Next Steps

### Immediate (Complete runtime.rs)
1. **Update remaining `cudarc` references** in runtime.rs (lines 115-405)
2. **Implement `Device` methods:**
   - `hip_stream()` - Return HIP stream
   - `hip_device()` - Return HIP device
   - `new()` - Create device from HIP device
3. **Implement `Slice` methods:**
   - Memory allocation
   - Host-to-device copy
   - Device-to-host copy
4. **Implement `ToSlice` trait** for type conversions

### Testing
1. **Add to workspace:** Update `/home/vince/Projects/rbee/deps/ug/Cargo.toml`
2. **Compile test:** `cargo check --manifest-path deps/ug/ug-rocm/Cargo.toml`
3. **Integration test:** Update Candle to use ug-rocm

### Integration with Candle
1. **Update Candle's Cargo.toml:**
   ```toml
   [dependencies]
   ug-rocm = { path = "../ug/ug-rocm", optional = true }
   
   [features]
   rocm = ["dep:rocm-rs", "dep:ug-rocm"]
   ```

2. **Implement `RocmDevice::compile()`:**
   ```rust
   pub fn compile(
       &self,
       func_name: &'static str,
       kernel: ug::lang::ssa::Kernel,
   ) -> Result<rocm_rs::hip::Function> {
       // Generate HIP code
       let mut buf = vec![];
       ug_rocm::code_gen::gen(&mut buf, func_name, &kernel)?;
       let hip_code = String::from_utf8(buf)?;
       
       // Compile using rocm-rs
       let module = rocm_rs::hip::module::compile_and_load(&hip_code, &[])?;
       let func = module.get_function(func_name)?;
       
       Ok(func)
   }
   ```

3. **Update `UgIOp1::new()`:**
   ```rust
   #[cfg(feature = "rocm")]
   {
       let device = device.as_rocm_device()?;
       let func = device.compile(name, kernel)?;
       Ok(Self { name, func })
   }
   ```

## Why This Approach Works

1. **HIP is CUDA-compatible** - Most CUDA code works with minimal changes
2. **rocm-rs provides compilation** - `compile_and_load()` handles hipcc
3. **Type mapping is straightforward** - Only BF16/F16 types differ
4. **Launch API is similar** - Grid/block dimensions work the same way

## Status Summary

✅ **Repository setup complete**  
✅ **Cargo.toml configured**  
✅ **lib.rs updated**  
✅ **code_gen.rs partially updated** (DType formatting done)  
✅ **runtime.rs partially updated** (structs and imports done)  
🚧 **runtime.rs implementation** (methods still needed)  
⏳ **Testing** (pending completion)  
⏳ **Candle integration** (pending completion)

## Estimated Completion

- **runtime.rs completion:** 2-3 hours
- **Testing:** 1 hour
- **Candle integration:** 1 hour
- **Total:** 4-5 hours of focused work

## References

- **CUDA version:** `/home/vince/Projects/rbee/deps/ug/ug-cuda/`
- **rocm-rs:** `/home/vince/Projects/rbee/deps/rocm-rs/`
- **Candle ROCm backend:** `/home/vince/Projects/rbee/deps/candle/candle-core/src/rocm_backend/`
- **Previous analysis:** `/home/vince/Projects/rbee/deps/candle/.docs/TEAM_510_CUSTOM_OP_ROCM_IMPLEMENTATION.md`
