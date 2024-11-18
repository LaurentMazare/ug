use crate::LB;
use ug::Result;
use ug_cuda::runtime::Slice;

const CAT_CU: &str = include_str!("cat.cu");
const ROPE_CU: &str = include_str!("rope.cu");
const ROPEI_CU: &str = include_str!("ropei.cu");
const SOFTMAX_CU: &str = include_str!("softmax.cu");

impl crate::Device for ug_cuda::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
        let func = src.device().compile_cu(ROPEI_CU, "ropei_f32", "ropei_f32", cfg)?;

        let (b, h, t, d) = src.shape().dims4()?;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            unsafe {
                func.launch8((src, cos, sin, pos, dst, (b * h) as u32, (t * d) as u32, d as u32))?
            };
            Ok(())
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
        let func = src.device().compile_cu(ROPE_CU, "rope_f32", "rope_f32", cfg)?;

        let (b, h, t, d) = src.shape().dims4()?;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            unsafe {
                func.launch8((src, cos, sin, pos, dst, (b * h) as u32, (t * d) as u32, d as u32))?
            };
            Ok(())
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
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
        let func = lhs.device().compile_cu(CAT_CU, "cat_f32", "cat_f32", cfg)?;

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
        let d1 = l_dims[..axis].iter().product::<usize>();
        let d2_l = l_dims[axis..].iter().product::<usize>();
        let d2_r = r_dims[axis..].iter().product::<usize>();
        let d2_lr = d2_l + d2_r;
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [lhs, rhs, dst]: [&mut Slice; 3] = vs.try_into().unwrap();
            unsafe {
                func.launch7((lhs, rhs, dst, d1 as u32, d2_l as u32, d2_r as u32, d2_lr as u32))?
            };
            Ok(())
        };
        LB::custom(f, vec![lhs.clone(), rhs.clone()], dst_dims, lhs.dtype(), lhs.device())
    }

    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(1);
        let func = src.device().compile_cu(SOFTMAX_CU, "softmax_f32", "softmax_f32", cfg)?;

        let rank = src.rank();
        let dim_m1 = src.dims()[rank - 1];
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [src, dst]: [&mut Slice; 2] = vs.try_into().unwrap();
            unsafe { func.launch3((src, dst, dim_m1 as i32))? };
            Ok(())
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn causal_mask(_: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("causal_mask is not yet implemented for the cuda backend")
    }
}