use crate::{Device, LB};
use rayon::prelude::*;
use ug::{CpuStorage, Result};

impl Device for ug::CpuDevice {
    fn custom_softmax(src: &LB<Self>) -> Result<LB<Self>> {
        let rank = src.rank();
        let dim_m1 = src.dims()[rank - 1];
        let span_sm = tracing::span!(tracing::Level::TRACE, "softmax");
        let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
            let _guard = span_sm.enter();
            let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
            let dst = dst.data_mut::<f32>()?;
            let src = src.data::<f32>()?;
            src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
                let mut max = f32::NEG_INFINITY;
                for &v in src.iter() {
                    max = f32::max(v, max)
                }
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = (*s - max).exp();
                }
                let sum_exp = dst.iter().sum::<f32>();
                for d in dst.iter_mut() {
                    *d /= sum_exp
                }
            });
            Ok(())
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn causal_mask(src: &LB<Self>) -> Result<LB<Self>> {
        let span = tracing::span!(tracing::Level::TRACE, "mask");
        let (_b_sz, _num_heads, s1, s2) = src.dims4()?;
        let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
            let _guard = span.enter();
            let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
            let dst = dst.data_mut::<f32>()?;
            let src = src.data::<f32>()?;
            src.par_chunks(s1 * s2).zip(dst.par_chunks_mut(s1 * s2)).for_each(|(src, dst)| {
                for i1 in 0..s1 {
                    for i2 in 0..=i1 {
                        let index = i1 * s2 + i2;
                        dst[index] = src[index];
                    }
                    for i2 in (i1 + 1)..s2 {
                        let index = i1 * s2 + i2;
                        dst[index] = f32::NEG_INFINITY;
                    }
                }
            });
            Ok(())
        };
        LB::custom(f, vec![src.clone()], src.shape(), src.dtype(), src.device())
    }

    fn rope_i(src: &LB<Self>, cos: &LB<Self>, sin: &LB<Self>, pos: &LB<Self>) -> Result<LB<Self>> {
        let (b, h, t, d) = src.shape().dims4()?;
        let span = tracing::span!(tracing::Level::TRACE, "ropei");
        let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
            let _guard = span.enter();
            let [src, cos, sin, pos, dst]: [&mut CpuStorage; 5] = vs.try_into().unwrap();
            let src = src.data::<f32>()?;
            let cos = cos.data::<f32>()?;
            let sin = sin.data::<f32>()?;
            // TODO: this only work for single positions.
            let pos = pos.data::<i32>()?[0] as usize;
            let dst = dst.data_mut::<f32>()?;
            let cos = &cos[pos * d / 2..];
            let sin = &sin[pos * d / 2..];
            dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
                for i_over_2 in 0..t * d / 2 {
                    let i = 2 * i_over_2;
                    let (s_i, s_ip) = (src[i], src[i + 1]);
                    dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                    dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
                }
            });
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
        let span = tracing::span!(tracing::Level::TRACE, "rope");
        let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
            let _guard = span.enter();
            let [src, cos, sin, pos, dst]: [&mut CpuStorage; 5] = vs.try_into().unwrap();
            let src = src.data::<f32>()?;
            let cos = cos.data::<f32>()?;
            let sin = sin.data::<f32>()?;
            // TODO: this only work for single positions.
            let pos = pos.data::<i32>()?[0] as usize;
            let dst = dst.data_mut::<f32>()?;
            let cos = &cos[pos * d / 2..];
            let sin = &sin[pos * d / 2..];
            dst.par_chunks_mut(t * d).zip(src.par_chunks(t * d)).for_each(|(dst, src)| {
                for i_t in 0..t {
                    for i_d in 0..d / 2 {
                        let i1 = i_t * d + i_d;
                        let i2 = i1 + d / 2;
                        let i_cs = i_t * (d / 2) + i_d;
                        let (src_i1, src_i2) = (src[i1], src[i2]);
                        dst[i1] = src_i1 * cos[i_cs] - src_i2 * sin[i_cs];
                        dst[i2] = src_i1 * sin[i_cs] + src_i2 * cos[i_cs];
                    }
                }
            });
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
        let span = tracing::span!(tracing::Level::TRACE, "cat");
        let l_dims = l_dims.to_vec();
        let r_dims = r_dims.to_vec();
        let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
            let _guard = span.enter();
            let [lhs, rhs, dst]: [&mut CpuStorage; 3] = vs.try_into().unwrap();
            let dst = dst.data_mut::<f32>()?;
            let lhs = lhs.data::<f32>()?;
            let rhs = rhs.data::<f32>()?;
            let d1 = l_dims[..axis].iter().product::<usize>();
            let d2_l = l_dims[axis..].iter().product::<usize>();
            let d2_r = r_dims[axis..].iter().product::<usize>();
            let d2_lr = d2_l + d2_r;
            for i1 in 0..d1 {
                let lhs = &lhs[i1 * d2_l..(i1 + 1) * d2_l];
                let rhs = &rhs[i1 * d2_r..(i1 + 1) * d2_r];
                let dst = &mut dst[i1 * d2_lr..(i1 + 1) * d2_lr];
                dst[..lhs.len()].copy_from_slice(lhs);
                dst[lhs.len()..].copy_from_slice(rhs);
            }
            Ok(())
        };
        LB::custom(f, vec![lhs.clone(), rhs.clone()], dst_dims, lhs.dtype(), lhs.device())
    }
}
