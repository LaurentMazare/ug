use crate::LB;
use ug::Result;

#[allow(unused)]
impl crate::Device for ug_cuda::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        todo!()
    }
    fn rope(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        todo!()
    }
    fn cat(lhs: &LB<Self>, rhs: &LB<Self>, axis: usize) -> Result<LB<Self>> {
        todo!()
    }
    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        todo!()
    }
    fn causal_mask(src: &LB<Self>) -> Result<LB<Self>> {
        todo!()
    }
}
