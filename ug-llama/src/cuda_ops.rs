use crate::LB;
use ug::Result;
use ug_cuda::runtime::Slice;

const ROPE_CU: &'static str = include_str!("rope.cu");

#[allow(unused)]
impl crate::Device for ug_cuda::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
        // TODO: Add rope_i on the cuda side
        let func = src.device().compile_cu(&ROPE_CU, "rope_f32", "rope_f32", cfg)?;

        let (b, h, t, d) = src.shape().dims4()?;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            ug::bail!("rope-i is not implemented for the cuda backend")
        };
        LB::custom(
            f,
            vec![src.clone(), cos.clone(), sin.clone(), pos.clone()],
            (b, h, t, d),
            src.dtype(),
            src.device(),
        )
    }

    fn rope(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
        let func = src.device().compile_cu(&ROPE_CU, "rope_f32", "rope_f32", cfg)?;

        let (b, h, t, d) = src.shape().dims4()?;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            ug::bail!("rope is not implemented for the cuda backend")
        };
        LB::custom(
            f,
            vec![src.clone(), cos.clone(), sin.clone(), pos.clone()],
            (b, h, t, d),
            src.dtype(),
            src.device(),
        )
    }

    fn cat(lhs: &LB<Self>, rhs: &LB<Self>, axis: usize) -> Result<LB<Self>> {
        let l_dims = lhs.dims();
        let r_dims = rhs.dims();
        if axis >= l_dims.len() {
            ug::bail!("unexpected axis {axis} for cat {l_dims:?}")
        }
        if l_dims.len() != r_dims.len() {
            ug::bail!("unexpected shapes for cat {l_dims:?} {r_dims:?} axis: {axis}")
        }
        for (i, (l, r)) in l_dims.iter().zip(r_dims.iter()).enumerate() {
            if axis != i && *l != *r {
                ug::bail!("unexpected shapes for cat {l_dims:?} {r_dims:?} axis: {axis}")
            }
        }
        let mut dst_dims = l_dims.to_vec();
        dst_dims[axis] = l_dims[axis] + r_dims[axis];
        let span = tracing::span!(tracing::Level::TRACE, "cat");
        let l_dims = l_dims.to_vec();
        let r_dims = r_dims.to_vec();
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            ug::bail!("cat is not implemented for the cuda backend")
        };
        LB::custom(f, vec![lhs.clone(), rhs.clone()], dst_dims, lhs.dtype(), lhs.device())
    }

    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        let rank = src.rank();
        let dim_m1 = src.dims()[rank - 1];
        let span_sm = tracing::span!(tracing::Level::TRACE, "softmax");
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            ug::bail!("custom_softmax is not implemented for the cuda backend")
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn causal_mask(src: &LB<Self>) -> Result<LB<Self>> {
        let (_b_sz, _num_heads, s1, s2) = src.dims4()?;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            ug::bail!("causal_mask is not implemented for the cuda backend")
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }
}
