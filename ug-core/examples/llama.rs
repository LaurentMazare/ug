// wget https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
#![allow(unused)]
use rayon::prelude::*;
use ug::{CpuDevice, CpuStorage, LazyBuffer, Result, Slice, WithDType};

type LB = LazyBuffer<CpuDevice>;

const UNK_TOKEN: u32 = 0;
const BOS_TOKEN: u32 = 1;
const EOS_TOKEN: u32 = 1;

fn index_select(src: &LB, ids: &[u32]) -> Result<LB> {
    let seq_len = ids.len();
    let (_, h) = src.shape().dims2()?;
    let out = LB::cst(0f32, (seq_len, h), &CpuDevice)?;
    let ids = ids.to_vec();
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, mut dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            dst[i * h..(i + 1) * h].copy_from_slice(&src[id * h..(id + 1) * h]);
        }
        Ok(())
    };
    let out = out.custom(std::sync::Arc::new(f), vec![src.clone()])?;
    Ok(out)
}

fn rms_norm(src: &LB, alpha: &LB, eps: f32) -> Result<LB> {
    let (d, dim_m1) = src.shape().dims2()?;
    let out = LB::cst(0f32, (d, dim_m1), &CpuDevice)?;
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, alpha, mut dst]: [&mut CpuStorage; 3] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        let alpha = alpha.data::<f32>()?;
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v * v).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = *s / m * *alpha
            }
        });
        Ok(())
    };
    let out = out.custom(std::sync::Arc::new(f), vec![src.clone(), alpha.clone()])?;
    Ok(out)
}

fn main() -> Result<()> {
    let st = unsafe { ug::safetensors::MmapedSafetensors::new("model.safetensors")? };
    let tensor = st.load_with_cast("model.embed_tokens.weight", ug::DType::F32, &CpuDevice)?;
    let tensor = index_select(&tensor, &[BOS_TOKEN])?;
    println!("{:?} {:?} {}", tensor.shape(), tensor.dtype(), tensor.realized());

    let schedule = ug::Schedule::create_one(&tensor)?;
    let schedule = schedule.compile()?;
    schedule.run()?;
    println!("{:?} {:?} {}", tensor.shape(), tensor.dtype(), tensor.realized());

    {
        let data = tensor.data().lock().unwrap();
        let data = data.as_ref().unwrap();
        let data = data.to_vec::<f32>()?;
        println!("{:?}", &data[..10]);
    };

    Ok(())
}
