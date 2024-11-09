// wget https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
#![allow(unused)]
use rayon::prelude::*;
use ug::{CpuDevice, CpuStorage, LazyBuffer, Result, Slice, WithDType};

type LB = LazyBuffer<CpuDevice>;
type ST = ug::safetensors::MmapedSafetensors;

const UNK_TOKEN: u32 = 0;
const BOS_TOKEN: u32 = 1;
const EOS_TOKEN: u32 = 1;

#[derive(Debug, Clone)]
pub enum HiddenAct {
    Silu,
}

#[derive(Debug, Clone)]
pub struct Config {
    hidden_act: HiddenAct,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    rms_norm_eps: f64,
    rope_interleaved: bool,
    rope_theta: f64,
    tie_word_embeddings: bool,
    vocab_size: usize,
}

impl Config {
    pub fn smollm2_135m() -> Self {
        Self {
            hidden_act: HiddenAct::Silu,
            hidden_size: 576,
            intermediate_size: 1536,
            max_position_embeddings: 8192,
            num_attention_heads: 9,
            num_hidden_layers: 30,
            num_key_value_heads: 3,
            rms_norm_eps: 1e-5,
            rope_interleaved: false,
            rope_theta: 1e5,
            tie_word_embeddings: true,
            vocab_size: 49152,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

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
    let out = out.custom(f, vec![src.clone()])?;
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
    let out = out.custom(f, vec![src.clone(), alpha.clone()])?;
    Ok(out)
}

fn rope_i(src: &LB, cos: &LB, sin: &LB, pos: usize) -> Result<LB> {
    let (b, h, t, d) = src.shape().dims4()?;
    let out = LB::cst(0f32, (b, h, t, d), &CpuDevice)?;
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, cos, sin, mut dst]: [&mut CpuStorage; 4] = vs.try_into().unwrap();
        let src = src.data::<f32>()?;
        let cos = cos.data::<f32>()?;
        let sin = sin.data::<f32>()?;
        let dst = dst.data_mut::<f32>()?;
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).for_each(|dst| {
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let (s_i, s_ip) = (dst[i], dst[i + 1]);
                dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
            }
        });
        Ok(())
    };
    let out = out.custom(f, vec![src.clone(), cos.clone(), sin.clone()])?;
    Ok(out)
}

fn rope(src: &LB, cos: &LB, sin: &LB, pos: usize) -> Result<LB> {
    let (b, h, t, d) = src.shape().dims4()?;
    let out = LB::cst(0f32, (b, h, t, d), &CpuDevice)?;
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, cos, sin, mut dst]: [&mut CpuStorage; 4] = vs.try_into().unwrap();
        let src = src.data::<f32>()?;
        let cos = cos.data::<f32>()?;
        let sin = sin.data::<f32>()?;
        let dst = dst.data_mut::<f32>()?;
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).for_each(|dst| {
            for i_t in 0..t {
                for i_d in 0..d / 2 {
                    let i1 = i_t * d + i_d;
                    let i2 = i1 + d / 2;
                    let i_cs = i_t * (d / 2) + i_d;
                    let (src_i1, src_i2) = (dst[i1], dst[i2]);
                    dst[i1] = src_i1 * cos[i_cs] - src_i2 * sin[i_cs];
                    dst[i2] = src_i1 * sin[i_cs] + src_i2 * cos[i_cs];
                }
            }
        });
        Ok(())
    };
    let out = out.custom(f, vec![src.clone(), cos.clone(), sin.clone()])?;
    Ok(out)
}

fn softmax(src: &LB) -> Result<LB> {
    let (d, dim_m1) = src.shape().dims2()?;
    let out = LB::cst(0f32, (d, dim_m1), &CpuDevice)?;
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, mut dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
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
    let out = out.custom(f, vec![src.clone()])?;
    Ok(out)
}

fn silu(src: &LB) -> Result<LB> {
    let out = LB::cst(0f32, src.shape(), &CpuDevice)?;
    let f = move |mut vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, mut dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s / (1. + f32::exp(-*s))
        }
        Ok(())
    };
    let out = out.custom(f, vec![src.clone()])?;
    Ok(out)
}

struct RmsNorm {
    alpha: LB,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, name: &str, st: &ST) -> Result<Self> {
        let alpha = st.load_with_cast(name, ug::DType::F32, &CpuDevice)?;
        if alpha.dims() != [dim] {
            ug::bail!("unexpected shape for {name}: {:?}, expected {dim}", alpha.shape())
        }
        Ok(Self { alpha, eps })
    }
}

struct Linear {
    w: LB,
    #[allow(unused)]
    in_c: usize,
    #[allow(unused)]
    out_c: usize,
}

impl Linear {
    fn new(in_c: usize, out_c: usize, name: &str, st: &ST) -> Result<Self> {
        let w = st.load_with_cast(name, ug::DType::F32, &CpuDevice)?;
        if w.dims() != [out_c, in_c] {
            ug::bail!("unexpected shape for {name}: {:?}, exp ({out_c}, {in_c})", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    head_dim: usize,
}

struct Layer {
    rms1: RmsNorm,
    attn: Attention,
    rms2: RmsNorm,
    mlp: Mlp,
}

struct Model {
    embedding: LB,
    layers: Vec<Layer>,
    ln_f: RmsNorm,
    lm_head: Linear,
    config: Config,
}

impl Model {
    fn new(cfg: &Config, st: &ug::safetensors::MmapedSafetensors) -> Result<Model> {
        let embedding =
            st.load_with_cast("model.embed_tokens.weight", ug::DType::F32, &CpuDevice)?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, "model.norm.weight", st)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear { w: embedding.clone(), in_c: cfg.hidden_size, out_c: cfg.vocab_size }
        } else {
            ug::bail!("tie_word_embeddings == false is not supported yet")
        };
        let i_sz = cfg.intermediate_size;
        let h_sz = cfg.hidden_size;
        let kv_sz = cfg.head_dim() * cfg.num_key_value_heads;
        let eps = cfg.rms_norm_eps;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let name = format!("model.layers.{layer_idx}");
            let rms1 = RmsNorm::new(h_sz, eps, &format!("{name}.input_layernorm.weight"), st)?;
            let rms2 =
                RmsNorm::new(h_sz, eps, &format!("{name}.post_attention_layernorm.weight"), st)?;
            let c_fc1 = Linear::new(h_sz, i_sz, &format!("{name}.mlp.gate_proj.weight"), st)?;
            let c_fc2 = Linear::new(h_sz, i_sz, &format!("{name}.mlp.up_proj.weight"), st)?;
            let c_proj = Linear::new(i_sz, h_sz, &format!("{name}.mlp.down_proj.weight"), st)?;
            let q_proj = Linear::new(h_sz, h_sz, &format!("{name}.self_attn.q_proj.weight"), st)?;
            let k_proj = Linear::new(h_sz, kv_sz, &format!("{name}.self_attn.k_proj.weight"), st)?;
            let v_proj = Linear::new(h_sz, kv_sz, &format!("{name}.self_attn.v_proj.weight"), st)?;
            let o_proj = Linear::new(h_sz, h_sz, &format!("{name}.self_attn.o_proj.weight"), st)?;
            let attn = Attention { q_proj, k_proj, v_proj, o_proj, head_dim: cfg.head_dim() };
            let mlp = Mlp { c_fc1, c_fc2, c_proj };
            layers.push(Layer { rms1, attn, rms2, mlp })
        }
        Ok(Self { embedding, layers, ln_f, lm_head, config: cfg.clone() })
    }
}

fn main() -> Result<()> {
    let st = unsafe { ug::safetensors::MmapedSafetensors::new("model.safetensors")? };
    let _model = Model::new(&Config::smollm2_135m(), &st)?;
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
