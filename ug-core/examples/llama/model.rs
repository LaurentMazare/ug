use rayon::prelude::*;
use ug::{CpuDevice, CpuStorage, DType, LazyBuffer, Result, D};

pub type LB = LazyBuffer<CpuDevice>;
type ST = ug::safetensors::MmapedSafetensors;

#[derive(serde::Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum HiddenAct {
    Silu,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub hidden_act: HiddenAct,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_interleaved: Option<bool>,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn index_select(src: &LB, ids: &LB) -> Result<LB> {
    use ug::lang::ssa::Instr as I;

    fn arg(index: usize, dtype: DType) -> I {
        I::DefineGlobal { index, dtype }
    }

    let dtype = DType::F32;
    let (b_sz, seq_len) = ids.dims2()?;
    let (_, h) = src.shape().dims2()?;

    let mut b = ug::block::Block::empty();
    {
        let src = b.push(arg(0, dtype)).to_varid();
        let ids = b.push(arg(1, DType::I32)).to_varid();
        let dst = b.push(arg(2, dtype)).to_varid();
        let r1 = b.range(0, (b_sz * seq_len) as i32);
        let src_off = b.push(I::Load { src: ids, offset: r1.id().to_a(), dtype });
        let src_off = b.mul(src_off, h as i32);
        let dst_off = b.mul(r1.id(), h as i32);
        let r2 = b.range(0, h as i32);
        let src_off = b.binary(ug::lang::BinaryOp::Add, src_off, r2.id(), DType::I32);
        let dst_off = b.binary(ug::lang::BinaryOp::Add, dst_off, r2.id(), DType::I32);
        let load_i = b.push(I::Load { src, offset: src_off.to_a(), dtype });
        b.push(I::Store { dst, offset: dst_off.to_a(), value: load_i.to_a(), dtype });
        b.end_range(r2)?;
        b.end_range(r1)?;
    }
    let instrs = b.relocate()?;
    let ssa = ug::lang::ssa::Kernel::from_instrs(instrs)?;
    LB::ssa(ssa, vec![src.clone(), ids.clone()], (b_sz, seq_len, h), src.dtype(), src.device())
}

fn rms_norm(src: &LB, alpha: &LB, eps: f32) -> Result<LB> {
    use ug::lang::{BinaryOp as B, ReduceOp as R, UnaryOp as U};
    let rank = src.rank();
    let dim_m1 = src.dims()[rank - 1];
    let sum2 = src.binary(B::Mul, src.clone())?.reduce(R::Sum, rank - 1)?;
    let s2s = sum2.shape();
    let m = sum2
        .binary(B::Mul, LB::cst(1f32 / dim_m1 as f32, s2s, &CpuDevice)?)?
        .binary(B::Add, LB::cst(eps, s2s, &CpuDevice)?)?
        .unary(U::Sqrt)?;
    src.binary(B::Div, m.broadcast(src.shape())?)?.binary(B::Mul, alpha.broadcast(src.shape())?)
}

fn rope_i(src: &LB, cos: &LB, sin: &LB, pos: &LB) -> Result<LB> {
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

fn rope(src: &LB, cos: &LB, sin: &LB, pos: &LB) -> Result<LB> {
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

fn cat(lhs: &LB, rhs: &LB, axis: usize) -> Result<LB> {
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
    let span = tracing::span!(tracing::Level::TRACE, "repeat");
    let f = move |vs: Vec<&mut CpuStorage>| -> Result<()> {
        let _guard = span.enter();
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
    LB::custom(f, vec![src.clone()], dst_dims, src.dtype(), src.device())
}

pub fn softmax(src: &LB) -> Result<LB> {
    let s = src.shape();
    let max = src.reduce(ug::lang::ReduceOp::Max, D::Minus1)?;
    let diff = src.binary(ug::lang::BinaryOp::Sub, max.broadcast(s)?)?;
    let exp = diff.unary(ug::lang::UnaryOp::Exp)?;
    let sum_exp = exp.reduce(ug::lang::ReduceOp::Sum, D::Minus1)?;
    exp.binary(ug::lang::BinaryOp::Div, sum_exp.broadcast(s)?)
}

pub fn custom_softmax(src: &LB) -> Result<LB> {
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

fn causal_mask(src: &LB) -> Result<LB> {
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

fn silu(src: &LB) -> Result<LB> {
    use ug::lang::{BinaryOp as B, UnaryOp as U};
    let exp_m = src.unary(U::Neg)?.unary(U::Exp)?;
    let one = LB::cst(1f32, (), &CpuDevice)?.broadcast(exp_m.shape())?;
    let den = exp_m.binary(B::Add, one)?;
    src.binary(B::Div, den)
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
        xs.matmul_t(self.w.clone())
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    hidden_act: HiddenAct,
}

impl Mlp {
    fn fwd(&self, xs: &LB) -> Result<LB> {
        let xs1 = self.c_fc1.fwd(xs)?;
        let xs2 = self.c_fc2.fwd(xs)?;
        let xs1 = match self.hidden_act {
            HiddenAct::Silu => silu(&xs1)?,
        };
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
    rope_interleaved: bool,
    custom_softmax: bool,
}

pub struct Cache {
    prev_k: LB,
    prev_v: LB,
}

impl Cache {
    pub fn new(cfg: &Config) -> Result<Vec<Self>> {
        let mut cache = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let prev_k =
                LB::cst(0f32, (1, cfg.num_key_value_heads, 0, cfg.head_dim()), &CpuDevice)?;
            let prev_v =
                LB::cst(0f32, (1, cfg.num_key_value_heads, 0, cfg.head_dim()), &CpuDevice)?;
            cache.push(Cache { prev_k, prev_v });
        }
        Ok(cache)
    }
}

impl Attention {
    // We use a mutable cache rather than returning an updated value. This makes the function
    // signatures slightly simpler but introduces more mutability.
    fn fwd(&self, xs: &LB, r: &Rope, pos: &LB, cache: &mut Cache) -> Result<LB> {
        let (b_sz, seq_len, _hidden_size) = xs.shape().dims3()?;
        let q = self.q_proj.fwd(xs)?;
        let k = self.k_proj.fwd(xs)?;
        let v = self.v_proj.fwd(xs)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let q = q.transpose(2, 1)?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = k.transpose(2, 1)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.transpose(2, 1)?;

        let q = if self.rope_interleaved {
            rope_i(&q, &r.cos, &r.sin, pos)?
        } else {
            rope(&q, &r.cos, &r.sin, pos)?
        };
        let k = if self.rope_interleaved {
            rope_i(&k, &r.cos, &r.sin, pos)?
        } else {
            rope(&k, &r.cos, &r.sin, pos)?
        };
        let k = cat(&cache.prev_k, &k, 2)?;
        let v = cat(&cache.prev_v, &v, 2)?;

        cache.prev_k = k.clone();
        cache.prev_v = v.clone();
        let k = repeat(&k, 1, self.num_heads / self.num_kv_heads)?;
        let v = repeat(&v, 1, self.num_heads / self.num_kv_heads)?;

        // attention
        let att = q.matmul_t(k)?;
        let scale = LB::cst((self.head_dim as f32).powf(-0.5), (), q.device())?;
        let scale = scale.broadcast(att.shape())?;
        let att = att.binary(ug::lang::BinaryOp::Mul, scale)?;
        let att = if seq_len == 1 { att } else { causal_mask(&att)? };
        let att = if self.custom_softmax { custom_softmax(&att)? } else { softmax(&att)? };
        let xs = att.matmul(v)?;

        // final proj
        let xs = xs.transpose(2, 1)?;
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
    fn fwd(&self, xs: &LB, rope: &Rope, pos: &LB, cache: &mut Cache) -> Result<LB> {
        let residual = xs.clone();
        let xs = self.rms1.fwd(xs)?;
        let xs = self.attn.fwd(&xs, rope, pos, cache)?;
        let xs = xs.binary(ug::lang::BinaryOp::Add, residual)?;
        let residual = xs.clone();
        let xs = self.rms2.fwd(&xs)?;
        let xs = self.mlp.fwd(&xs)?;
        let xs = xs.binary(ug::lang::BinaryOp::Add, residual)?;
        Ok(xs)
    }
}

struct Rope {
    cos: LB,
    sin: LB,
}

impl Rope {
    fn new(cfg: &Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = LB::from_slice(theta.into(), (1, head_dim / 2))?;
        let idx_theta = LB::from_slice(
            (0..max_seq_len).map(|v| v as f32).collect::<Vec<_>>().into(),
            (max_seq_len, 1),
        )?;
        let mm = idx_theta.matmul(theta)?;
        let cos = mm.unary(ug::lang::UnaryOp::Cos)?;
        let sin = mm.unary(ug::lang::UnaryOp::Sin)?;
        Ok(Self { cos, sin })
    }
}

pub struct Model {
    embedding: LB,
    rope: Rope,
    layers: Vec<Layer>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(
        cfg: &Config,
        custom_softmax: bool,
        st: &ug::safetensors::MmapedSafetensors,
    ) -> Result<Model> {
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
                rope_interleaved: cfg.rope_interleaved.unwrap_or(false),
                custom_softmax,
            };
            let mlp = Mlp { c_fc1, c_fc2, c_proj, hidden_act: cfg.hidden_act };
            layers.push(Layer { rms1, attn, rms2, mlp })
        }
        let rope = Rope::new(cfg)?;
        Ok(Self { embedding, layers, ln_f, lm_head, rope })
    }

    pub fn fwd(&self, tokens: &LB, pos: &LB, cache: &mut [Cache]) -> Result<LB> {
        let mut xs = index_select(&self.embedding, tokens)?;
        for (layer, cache) in self.layers.iter().zip(cache) {
            xs = layer.fwd(&xs, &self.rope, pos, cache)?
        }
        let xs = self.ln_f.fwd(&xs)?;
        let xs = self.lm_head.fwd(&xs)?;
        Ok(xs)
    }
}
