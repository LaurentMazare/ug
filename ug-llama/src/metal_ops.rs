use crate::LB;
use std::sync::OnceLock;
use ug::{lang::LaunchConfig, Result};
use ug_metal::runtime::{Func, Slice};

const CAT_M: &str = include_str!("cat.metal");
const ROPE_M: &str = include_str!("rope.metal");
const ROPEI_M: &str = include_str!("ropei.metal");
const SOFTMAX_M: &str = include_str!("softmax.metal");

impl crate::Device for ug_metal::runtime::Device {
    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        static ROPEI: OnceLock<Func> = OnceLock::new();
        let device = src.device();
        let (b, h, t, d) = src.shape().dims4()?;
        let cfg = LaunchConfig::for_num_elems((b * h * t * d) as u32 / 2);
        // TODO: Use get_or_try_init when available.
        let func = ROPEI.get_or_init(|| device.compile_metal(ROPEI_M, "ropei_f32", cfg).unwrap());

        let device = device.clone();
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            let cb = device.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let pl = func.pipeline()?;
            encoder.set_compute_pipeline_state(&pl);
            ug_metal::set_params!(
                encoder,
                (&*src, &*cos, &*sin, &*pos, &*dst, (b * h) as u32, (t * d) as u32, d as u32)
            );
            let grid_size = metal::MTLSize::new(cfg.grid_dim as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(cfg.block_dim as u64, 1, 1);
            encoder.use_resource(src.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(cos.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(sin.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(pos.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(dst.buffer(), metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
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
        static ROPE: OnceLock<Func> = OnceLock::new();
        let device = src.device();
        let (b, h, t, d) = src.shape().dims4()?;
        let cfg = LaunchConfig::for_num_elems((b * h * t * d) as u32 / 2);
        // TODO: Use get_or_try_init when available.
        let func = ROPE.get_or_init(|| device.compile_metal(ROPE_M, "rope_f32", cfg).unwrap());

        let device = device.clone();
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            // TODO: check the dtypes.
            let [src, cos, sin, pos, dst]: [&mut Slice; 5] = vs.try_into().unwrap();
            let cb = device.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let pl = func.pipeline()?;
            encoder.set_compute_pipeline_state(&pl);
            ug_metal::set_params!(
                encoder,
                (&*src, &*cos, &*sin, &*pos, &*dst, (b * h) as u32, (t * d) as u32, d as u32)
            );
            let grid_size = metal::MTLSize::new(cfg.grid_dim as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(cfg.block_dim as u64, 1, 1);
            encoder.use_resource(src.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(cos.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(sin.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(pos.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(dst.buffer(), metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
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
        static CAT: OnceLock<Func> = OnceLock::new();
        let device = lhs.device();
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
        let cfg = ug::lang::LaunchConfig { grid_dim: d1 as u32, block_dim: 32, shared_mem: 0 };
        // TODO: Use get_or_try_init when available.
        let func = CAT.get_or_init(|| lhs.device().compile_metal(CAT_M, "cat_f32", cfg).unwrap());
        let device = device.clone();
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [lhs, rhs, dst]: [&mut Slice; 3] = vs.try_into().unwrap();
            let cb = device.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let pl = func.pipeline()?;
            encoder.set_compute_pipeline_state(&pl);
            ug_metal::set_params!(
                encoder,
                (&*lhs, &*rhs, &*dst, d1 as u32, d2_l as u32, d2_r as u32, d2_lr as u32)
            );
            let grid_size = metal::MTLSize::new(cfg.grid_dim as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(cfg.block_dim as u64, 1, 1);
            encoder.use_resource(lhs.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(rhs.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(dst.buffer(), metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            Ok(())
        };
        LB::custom(f, vec![lhs.clone(), rhs.clone()], dst_dims, lhs.dtype(), lhs.device())
    }

    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        static CUSTOM_SOFTMAX: OnceLock<Func> = OnceLock::new();
        let device = src.device();
        let rank = src.rank();
        let dim_m1 = src.dims()[rank - 1];
        let num_elements = src.shape().num_elements();
        let n_rows = num_elements / dim_m1;
        let cfg = LaunchConfig { grid_dim: n_rows as u32, block_dim: 32, shared_mem: 0 };

        // TODO: Use get_or_try_init when available.
        let func = CUSTOM_SOFTMAX
            .get_or_init(|| device.compile_metal(SOFTMAX_M, "softmax_f32", cfg).unwrap());
        let device = device.clone();
        let f = move |vs: Vec<&mut Slice>| -> Result<()> {
            let [src, dst]: [&mut Slice; 2] = vs.try_into().unwrap();
            let cb = device.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let pl = func.pipeline()?;
            encoder.set_compute_pipeline_state(&pl);
            ug_metal::set_params!(encoder, (num_elements as u32, dim_m1 as u32, &*src, &*dst));
            let grid_size = metal::MTLSize::new(cfg.grid_dim as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(cfg.block_dim as u64, 1, 1);
            encoder.use_resource(src.buffer(), metal::MTLResourceUsage::Read);
            encoder.use_resource(dst.buffer(), metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            Ok(())
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn causal_mask(_: &LB<Self>) -> Result<LB<Self>> {
        ug::bail!("causal_mask is not yet implemented for the cuda backend")
    }
}
