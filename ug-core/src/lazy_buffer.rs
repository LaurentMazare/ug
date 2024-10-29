use crate::dtype::WithDType;
use crate::Result;

pub trait Device: Clone {
    type Slice<D: WithDType>;

    fn allocate_uninit<D: WithDType>(&self, len: usize) -> Result<Self::Slice<D>>;
    fn copy_host_to_device<D: WithDType>(src: &[D], dst: &mut Self::Slice<D>) -> Result<()>;
    fn copy_device_to_host<D: WithDType>(src: &Self::Slice<D>, dst: &mut [D]) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
}

pub struct LazyBuffer<D: Device, DT: WithDType> {
    data: Option<D::Slice<DT>>,
    device: D,
}

impl<D: Device, DT: WithDType> LazyBuffer<D, DT> {
    pub fn realized(&self) -> bool {
        self.data.is_some()
    }

    pub fn data(&self) -> Option<&D::Slice<DT>> {
        self.data.as_ref()
    }

    pub fn device(&self) -> &D {
        &self.device
    }
}
