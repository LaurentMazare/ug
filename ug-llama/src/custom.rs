use ug::{CpuDevice, CpuStorage, LazyBuffer, Result};

#[allow(unused)]
pub type LB = LazyBuffer<CpuDevice>;

pub fn _transpose(src: &LB, dim1: usize, dim2: usize) -> Result<LB> {
    let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
    if dim1 == dim2 {
        return Ok(src.clone());
    }
    let dims = src.dims().to_vec();
    if dim1 >= dims.len() || dim2 >= dims.len() {
        ug::bail!("unexpected dims ({dim1}, {dim2}) for transpose {dims:?}")
    }
    let mut dst_dims = dims.clone();
    dst_dims.swap(dim1, dim2);
    let span = tracing::span!(tracing::Level::TRACE, "transpose");
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let _guard = span.enter();
        let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        let d_i = dims[..dim1].iter().product::<usize>();
        let d_j = dims[dim1 + 1..dim2].iter().product::<usize>();
        let d_k = dims[(dim2 + 1)..].iter().product::<usize>();
        let d1 = dims[dim1];
        let d2 = dims[dim2];
        // Inefficient, we should blit the data where possible.
        // i: pre
        for i in 0..d_i {
            let src_idx = i * d1;
            let dst_idx = i * d2;
            for a1 in 0..d1 {
                let src_idx = (src_idx + a1) * d_j;
                // j: mid
                for j in 0..d_j {
                    let src_idx = (src_idx + j) * d2;
                    for a2 in 0..d2 {
                        let src_idx = (src_idx + a2) * d_k;
                        let dst_idx = ((((dst_idx + a2) * d_j) + j) * d1 + a1) * d_k;
                        dst[dst_idx..dst_idx + d_k].copy_from_slice(&src[src_idx..src_idx + d_k])
                    }
                }
            }
        }
        Ok(())
    };
    LB::custom(f, vec![src.clone()], dst_dims, src.dtype(), src.device())
}

pub fn _index_select(src: &LB, ids: &LB) -> Result<LB> {
    let (b_sz, seq_len) = ids.dims2()?;
    let (_, h) = src.shape().dims2()?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, ids, dst]: [&mut CpuStorage; 3] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        let ids = ids.data::<i32>()?;
        for (dst, ids) in dst.chunks_mut(seq_len * h).zip(ids.chunks(seq_len)) {
            for (i, id) in ids.iter().enumerate() {
                let id = *id as usize;
                dst[i * h..(i + 1) * h].copy_from_slice(&src[id * h..(id + 1) * h]);
            }
        }
        Ok(())
    };
    LB::custom(f, vec![src.clone(), ids.clone()], (b_sz, seq_len, h), src.dtype(), src.device())
}
