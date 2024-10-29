use crate::dtype::WithDType;

pub trait Device: Clone {
    type Slice<D: WithDType>;
}

pub struct LazyBuffer<D: Device, DT: WithDType> {
    data: Option<D::Slice<DT>>,
    device: D,
}

impl<D: Device, DT: WithDType> LazyBuffer<D, DT> {
    pub fn realized(&self) -> bool {
        self.data.is_some()
    }

    pub fn device(&self) -> &D {
        &self.device
    }
}
