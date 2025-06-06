use crate::LB;
use ug::Result;
use ug_cuda::cudarc::driver::PushKernelArg;
use ug_cuda::runtime::{CudaFunction, Func, LaunchConfig, Slice, WithErr};

const CAT_CU: &str = include_str!("cat.cu");
const ROPE_CU: &str = include_str!("rope.cu");
const ROPEI_CU: &str = include_str!("ropei.cu");
const SOFTMAX_CU: &str = include_str!("softmax.cu");

use std::sync::OnceLock;
static ROPEI: OnceLock<CudaFunction> = OnceLock::new();
static ROPE: OnceLock<CudaFunction> = OnceLock::new();
static CAT: OnceLock<CudaFunction> = OnceLock::new();
static CUSTOM_SOFTMAX: OnceLock<CudaFunction> = OnceLock::new();

impl crate::Device for ug_cuda::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        let (b, h, t, d) = src.shape().dims4()?;
        let cfg = LaunchConfig::for_num_elems((b * h * t * d) as u32 / 2);
        // TODO: Use get_or_try_init when available.
        let func = ROPEI.get_or_init(|| src.device().compile_cu(ROPEI_CU, "ropei_f32").unwrap());
        let stream = src.device().cudarc_stream().clone();
        let func = Func::new(src.device(), func.clone(), cfg);

        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            let mut builder = func.builder();
            let (src, _guard) = src.device_ptr_mut(&stream);
            let (cos, _guard) = cos.device_ptr_mut(&stream);
            let (sin, _guard) = sin.device_ptr_mut(&stream);
            let (pos, _guard) = pos.device_ptr_mut(&stream);
            let (dst, _guard) = dst.device_ptr_mut(&stream);
            ug_cuda::bargs!(
                builder,
                src,
                cos,
                sin,
                pos,
                dst,
                (b * h) as u32,
                (t * d) as u32,
                d as u32
            );
            unsafe { builder.launch(*func.launch_cfg()).w()? };
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
        let (b, h, t, d) = src.shape().dims4()?;
        let cfg = LaunchConfig::for_num_elems((b * h * t * d) as u32 / 2);
        // TODO: Use get_or_try_init when available.
        let func = ROPE.get_or_init(|| src.device().compile_cu(ROPE_CU, "rope_f32").unwrap());
        let stream = src.device().cudarc_stream().clone();
        let func = Func::new(src.device(), func.clone(), cfg);

        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            let mut builder = func.builder();
            let (src, _guard) = src.device_ptr_mut(&stream);
            let (cos, _guard) = cos.device_ptr_mut(&stream);
            let (sin, _guard) = sin.device_ptr_mut(&stream);
            let (pos, _guard) = pos.device_ptr_mut(&stream);
            let (dst, _guard) = dst.device_ptr_mut(&stream);
            ug_cuda::bargs!(
                builder,
                src,
                cos,
                sin,
                pos,
                dst,
                (b * h) as u32,
                (t * d) as u32,
                d as u32
            );
            unsafe { builder.launch(*func.launch_cfg()).w()? };
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
        let cfg = LaunchConfig {
            grid_dim: (d1 as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // TODO: Use get_or_try_init when available.
        let func = CAT.get_or_init(|| lhs.device().compile_cu(CAT_CU, "cat_f32").unwrap());
        let stream = lhs.device().cudarc_stream().clone();
        let func = Func::new(lhs.device(), func.clone(), cfg);
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [lhs, rhs, dst]: [&mut Slice; 3] = vs.try_into().unwrap();
            let (lhs, _guard) = lhs.device_ptr_mut(&stream);
            let (rhs, _guard) = rhs.device_ptr_mut(&stream);
            let (dst, _guard) = dst.device_ptr_mut(&stream);
            let mut builder = func.builder();
            ug_cuda::bargs!(
                builder,
                lhs,
                rhs,
                dst,
                d1 as u32,
                d2_l as u32,
                d2_r as u32,
                d2_lr as u32
            );
            unsafe { builder.launch(*func.launch_cfg()).w()? };
            Ok(())
        };
        LB::custom(f, vec![lhs.clone(), rhs.clone()], dst_dims, lhs.dtype(), lhs.device())
    }

    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        let rank = src.rank();
        let dim_m1 = src.dims()[rank - 1];
        let n_rows = src.shape().num_elements() / dim_m1;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (1, 32, 1),
            shared_mem_bytes: 0,
        };

        // TODO: Use get_or_try_init when available.
        let func = CUSTOM_SOFTMAX
            .get_or_init(|| src.device().compile_cu(SOFTMAX_CU, "softmax_f32").unwrap());
        let stream = src.device().cudarc_stream().clone();
        let func = Func::new(src.device(), func.clone(), cfg);
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [src, dst]: [&mut Slice; 2] = vs.try_into().unwrap();
            let (src, _guard) = src.device_ptr_mut(&stream);
            let (dst, _guard) = dst.device_ptr_mut(&stream);
            let mut builder = func.builder();
            ug_cuda::bargs!(builder, src, dst, dim_m1 as i32);
            unsafe { builder.launch(*func.launch_cfg()).w()? };
            Ok(())
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn causal_mask(_: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("causal_mask is not yet implemented for the cuda backend")
    }
}
