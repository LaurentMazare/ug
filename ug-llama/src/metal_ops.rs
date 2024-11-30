#![allow(unused)]

use crate::LB;
use ug::Result;
impl crate::Device for ug_metal::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("rope_i is not yet implemented for the metal backend")
    }

    fn rope(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("rope is not yet implemented for the metal backend")
    }

    fn cat(lhs: &LB<Self>, rhs: &LB<Self>, axis: usize) -> Result<LB<Self>> {
        ug::bail!("cat is not yet implemented for the metal backend")
    }

    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("custom_softmax is not yet implemented for the metal backend")
    }

    fn causal_mask(_: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("causal_mask is not yet implemented for the metal backend")
    }
}
