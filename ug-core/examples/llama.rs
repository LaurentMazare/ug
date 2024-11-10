// wget https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
use rayon::prelude::*;
use ug::{CpuDevice, CpuStorage, LazyBuffer, Result, Slice};

type LB = LazyBuffer<CpuDevice>;
type ST = ug::safetensors::MmapedSafetensors;

const NUM_HIDDEN_LAYERS: Option<usize> = Some(1);
#[allow(unused)]
const UNK_TOKEN: u32 = 0;
const BOS_TOKEN: u32 = 1;
#[allow(unused)]
const EOS_TOKEN: u32 = 2;

#[derive(Debug, Clone)]
pub enum HiddenAct {
    Silu,
}

#[allow(unused)]
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
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
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
    let rank = src.rank();
    let dim_m1 = src.dims()[rank - 1];
    let out = LB::cst(0f32, src.shape(), &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, alpha, dst]: [&mut CpuStorage; 3] = vs.try_into().unwrap();
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

#[allow(unused)]
fn rope_i(src: &LB, cos: &LB, sin: &LB, pos: usize) -> Result<LB> {
    let (b, h, t, d) = src.shape().dims4()?;
    let out = LB::cst(0f32, (b, h, t, d), &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, cos, sin, dst]: [&mut CpuStorage; 4] = vs.try_into().unwrap();
        let src = src.data::<f32>()?;
        let cos = cos.data::<f32>()?;
        let sin = sin.data::<f32>()?;
        let dst = dst.data_mut::<f32>()?;
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        dst.par_chunks_mut(t * d).for_each(|dst| {
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let (s_i, s_ip) = (src[i], src[i + 1]);
                dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
            }
        });
        Ok(())
    };
    let out = out.custom(f, vec![src.clone(), cos.clone(), sin.clone()])?;
    Ok(out)
}

#[allow(unused)]
fn rope(src: &LB, cos: &LB, sin: &LB, pos: usize) -> Result<LB> {
    let (b, h, t, d) = src.shape().dims4()?;
    let out = LB::cst(0f32, (b, h, t, d), &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, cos, sin, dst]: [&mut CpuStorage; 4] = vs.try_into().unwrap();
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
                    let (src_i1, src_i2) = (src[i1], src[i2]);
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

fn repeat(src: &LB, axis: usize, n_rep: usize) -> Result<LB> {
    if n_rep == 1 {
        return Ok(src.clone());
    }
    let dims = src.dims().to_vec();
    if axis >= dims.len() {
        ug::bail!("unexpected axis {axis} for repeat {dims:?}")
    }
    let mut dst_dims = dims.clone();
    dst_dims[axis] *= n_rep;
    let out = LB::cst(0f32, dst_dims, &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
        let dst = dst.data_mut::<f32>()?;
        let src = src.data::<f32>()?;
        let d_i = dims[..(axis + 1)].iter().product::<usize>();
        let d_j = dims[(axis + 1)..].iter().product::<usize>();
        for i in 0..d_i {
            let src = &src[i * d_j..(i + 1) * d_j];
            let dst = &mut dst[i * d_j * n_rep..(i + 1) * d_j * n_rep];
            for i_rep in 0..n_rep {
                dst[i_rep * d_j..(i_rep + 1) * d_j].copy_from_slice(src)
            }
        }
        Ok(())
    };
    let out = out.custom(f, vec![src.clone()])?;
    Ok(out)
}

fn transpose(src: &LB, dim1: usize, dim2: usize) -> Result<LB> {
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
    let out = LB::cst(0f32, dst_dims, &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
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
            for a1 in 0..d1 {
                // j: mid
                for j in 0..d_j {
                    for a2 in 0..d2 {
                        // k: post
                        for k in 0..d_k {
                            let src_idx = i * d1 * d_j * d2 * d_k
                                + a1 * d_j * d2 * d_k
                                + j * d2 * d_k
                                + a2 * d_k
                                + k;
                            let dst_idx = i * d2 * d_j * d1 * d_k
                                + a2 * d_j * d1 * d_k
                                + j * d1 * d_k
                                + a1 * d_k
                                + k;
                            dst[dst_idx] = src[src_idx]
                        }
                    }
                }
            }
        }
        Ok(())
    };
    let out = out.custom(f, vec![src.clone()])?;
    Ok(out)
}

fn softmax(src: &LB) -> Result<LB> {
    let rank = src.rank();
    let dim_m1 = src.dims()[rank - 1];
    let out = LB::cst(0f32, src.shape(), &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
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
    let out = out.custom(f, vec![src.clone()])?;
    Ok(out)
}

fn silu(src: &LB) -> Result<LB> {
    let out = LB::cst(0f32, src.shape(), &CpuDevice)?;
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let [src, dst]: [&mut CpuStorage; 2] = vs.try_into().unwrap();
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

    fn fwd(&self, xs: &LB) -> Result<LB> {
        rms_norm(xs, &self.alpha, self.eps as f32)
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

    fn fwd(&self, xs: &LB) -> Result<LB> {
        let r = self.w.rank();
        let w_t = transpose(&self.w, r - 2, r - 1)?;
        let w_t = w_t.reshape((1, self.in_c, self.out_c))?;
        xs.matmul(w_t)
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn fwd(&self, xs: &LB) -> Result<LB> {
        let xs1 = self.c_fc1.fwd(xs)?;
        let xs2 = self.c_fc2.fwd(xs)?;
        let xs1 = silu(&xs1)?;
        let xs = xs1.binary(ug::lang::BinaryOp::Mul, xs2)?;
        self.c_proj.fwd(&xs)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn fwd(&self, xs: &LB) -> Result<LB> {
        let (b_sz, seq_len, _hidden_size) = xs.shape().dims3()?;
        if seq_len != 1 {
            ug::bail!("seq_len {seq_len} > 1 is not supported as no causal mask is applied")
        }
        let q = self.q_proj.fwd(xs)?;
        let k = self.k_proj.fwd(xs)?;
        let v = self.v_proj.fwd(xs)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let q = transpose(&q, 2, 1)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = transpose(&k, 2, 1)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = transpose(&v, 2, 1)?;

        // TODO: Rope
        // TODO: KV Cache
        let k = repeat(&k, 1, self.num_heads / self.num_kv_heads)?;
        let v = repeat(&v, 1, self.num_heads / self.num_kv_heads)?;

        // attention
        let k = transpose(&k, 3, 2)?;
        let att = q.matmul(k)?;
        let scale =
            ug::LazyBuffer::cst((self.head_dim as f32).powf(-0.5), att.shape(), q.device())?;
        // TODO: There is a bug when using broadcasting before mul.
        let att = att.binary(ug::lang::BinaryOp::Mul, scale)?;
        let att = softmax(&att)?;
        let xs = att.matmul(v)?;

        // final proj
        let xs = transpose(&xs, 2, 1)?;
        let xs = xs.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        let xs = self.o_proj.fwd(&xs)?;
        Ok(xs)
    }
}

struct Layer {
    rms1: RmsNorm,
    attn: Attention,
    rms2: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn fwd(&self, xs: &LB) -> Result<LB> {
        let residual = xs.clone();
        let xs = self.rms1.fwd(xs)?;
        let xs = self.attn.fwd(&xs)?;
        let xs = xs.binary(ug::lang::BinaryOp::Add, residual)?;
        let residual = xs.clone();
        let xs = self.rms2.fwd(&xs)?;
        let xs = self.mlp.fwd(&xs)?;
        let xs = xs.binary(ug::lang::BinaryOp::Add, residual)?;
        Ok(xs)
    }
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
            let attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                head_dim: cfg.head_dim(),
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
            };
            let mlp = Mlp { c_fc1, c_fc2, c_proj };
            layers.push(Layer { rms1, attn, rms2, mlp })
        }
        Ok(Self { embedding, layers, ln_f, lm_head, config: cfg.clone() })
    }

    fn fwd(&self, tokens: &[u32]) -> Result<LB> {
        let seq_len = tokens.len();
        let xs = index_select(&self.embedding, tokens)?;
        let mut xs = xs.reshape((1, seq_len, self.config.hidden_size))?;
        for layer in self.layers.iter() {
            xs = layer.fwd(&xs)?
        }
        let xs = self.ln_f.fwd(&xs)?;
        let xs = self.lm_head.fwd(&xs)?;
        Ok(xs)
    }
}

fn main() -> Result<()> {
    let st = unsafe { ug::safetensors::MmapedSafetensors::new("model.safetensors")? };
    let mut config = Config::smollm2_135m();
    if let Some(num_hidden_layers) = NUM_HIDDEN_LAYERS {
        println!("overriding num layers: {num_hidden_layers}");
        config.num_hidden_layers = num_hidden_layers;
    }
    let model = Model::new(&config, &st)?;
    let tensor = model.fwd(&[BOS_TOKEN])?;
    println!("{:?} {:?} {}", tensor.shape(), tensor.dtype(), tensor.realized());
    let start_time = std::time::Instant::now();
    let schedule = ug::Schedule::create_one(&tensor)?;
    println!("schedule generated in {:.2}s", start_time.elapsed().as_secs_f32());
    let start_time = std::time::Instant::now();
    let schedule = schedule.compile()?;
    println!("schedule generated in {:.2}s", start_time.elapsed().as_secs_f32());
    schedule.run()?;
    println!("{:?} {:?} {}", tensor.shape(), tensor.dtype(), tensor.realized());

    {
        let data = tensor.data().lock().unwrap();
        let data = data.as_ref().unwrap();
        let data = data.to_vec::<f32>()?;
        println!("{} {:?}", data.len(), &data[..10]);
    };

    Ok(())
}
