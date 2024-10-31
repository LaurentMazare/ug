pub use crate::dtype::CpuStorage;
use crate::{DType, Result};
use half::{bf16, f16};

#[derive(Clone, Copy, Debug)]
pub struct CpuDevice;

impl crate::Device for CpuDevice {
    type Slice = CpuStorage;

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

    fn to_vec<DT: crate::WithDType>(&self) -> Result<Vec<DT>> {
        todo!()
    }

    fn copy_host_to_device<DT: crate::WithDType>(&mut self, _src: &[DT]) -> Result<()> {
        todo!()
    }

    fn copy_device_to_host<DT: crate::WithDType>(&self, _dst: &mut [DT]) -> Result<()> {
        todo!()
    }
}
