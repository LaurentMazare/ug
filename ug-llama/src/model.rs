use crate::{Device, LB, ST};
use ug::{DType, Result};

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

fn index_select<D: Device>(src: &LB<D>, ids: &LB<D>) -> Result<LB<D>> {
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
        let r1 = b.range(0, (b_sz * seq_len) as i32, 1);
        let src_off = b.push(I::Load { src: ids, offset: r1.id().to_a(), dtype });
        let src_off = b.mul(src_off, h as i32);
        let dst_off = b.mul(r1.id(), h as i32);
        let r2 = b.range(0, h as i32, 1);
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

fn rms_norm<D: Device>(src: &LB<D>, alpha: &LB<D>, eps: f32) -> Result<LB<D>> {
    use ug::lang::{BinaryOp as B, ReduceOp as R, UnaryOp as U};

    let device = src.device();
    let rank = src.rank();
    let dim_m1 = src.dims()[rank - 1];
    let sum2 = src.binary(B::Mul, src.clone())?.reduce(R::Sum, rank - 1)?;
    let s2s = sum2.shape();
    let m = sum2
        .binary(B::Mul, LB::cst(1f32 / dim_m1 as f32, s2s, device)?)?
        .binary(B::Add, LB::cst(eps, s2s, device)?)?
        .unary(U::Sqrt)?;
    src.binary(B::Div, m.broadcast(src.shape())?)?.binary(B::Mul, alpha.broadcast(src.shape())?)
}

fn repeat<D: Device>(lb: &LB<D>, axis: usize, n_rep: usize) -> Result<LB<D>> {
    if n_rep == 1 {
        return Ok(lb.clone());
    }
    let dims = lb.dims();
    if axis >= dims.len() {
        ug::bail!("unexpected axis {axis} for repeat {dims:?}")
    }
    let lb = lb.split_dim(axis, dims[axis], 1)?;
    let mut dims = lb.dims().to_vec();
    dims[axis + 1] = n_rep;
    lb.broadcast(dims)?.merge_dims(axis)
}

pub fn softmax<D: Device>(src: &LB<D>) -> Result<LB<D>> {
    let s = src.shape();
    let max = src.reduce(ug::lang::ReduceOp::Max, ug::D::Minus1)?;
    let diff = src.binary(ug::lang::BinaryOp::Sub, max.broadcast(s)?)?;
    let exp = diff.unary(ug::lang::UnaryOp::Exp)?;
    let sum_exp = exp.reduce(ug::lang::ReduceOp::Sum, ug::D::Minus1)?;
    exp.binary(ug::lang::BinaryOp::Div, sum_exp.broadcast(s)?)
}

fn silu<D: Device>(src: &LB<D>) -> Result<LB<D>> {
    use ug::lang::{BinaryOp as B, UnaryOp as U};
    let exp_m = src.unary(U::Neg)?.unary(U::Exp)?;
    let one = LB::cst(1f32, (), src.device())?.broadcast(exp_m.shape())?;
    let den = exp_m.binary(B::Add, one)?;
    src.binary(B::Div, den)
}

struct RmsNorm<D: Device> {
    alpha: LB<D>,
    eps: f64,
}

impl<D: Device> RmsNorm<D> {
    fn new(dim: usize, eps: f64, name: &str, st: &ST, device: &D) -> Result<Self> {
        let alpha = st.load_with_cast(name, ug::DType::F32, device)?;
        if alpha.dims() != [dim] {
            ug::bail!("unexpected shape for {name}: {:?}, expected {dim}", alpha.shape())
        }
        Ok(Self { alpha, eps })
    }

    fn fwd(&self, xs: &LB<D>) -> Result<LB<D>> {
        rms_norm(xs, &self.alpha, self.eps as f32)
    }
}

struct Linear<D: Device> {
    w: LB<D>,
    #[allow(unused)]
    in_c: usize,
    #[allow(unused)]
    out_c: usize,
}

impl<D: Device> Linear<D> {
    fn new(in_c: usize, out_c: usize, name: &str, st: &ST, device: &D) -> Result<Self> {
        let w = st.load_with_cast(name, ug::DType::F32, device)?;
        if w.dims() != [out_c, in_c] {
            ug::bail!("unexpected shape for {name}: {:?}, exp ({out_c}, {in_c})", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }

    fn fwd(&self, xs: &LB<D>) -> Result<LB<D>> {
        xs.matmul_t(self.w.clone())
    }
}

struct Mlp<D: Device> {
    c_fc1: Linear<D>,
    c_fc2: Linear<D>,
    c_proj: Linear<D>,
    hidden_act: HiddenAct,
}

impl<D: Device> Mlp<D> {
    fn fwd(&self, xs: &LB<D>) -> Result<LB<D>> {
        let xs1 = self.c_fc1.fwd(xs)?;
        let xs2 = self.c_fc2.fwd(xs)?;
        let xs1 = match self.hidden_act {
            HiddenAct::Silu => silu(&xs1)?,
        };
        let xs = xs1.binary(ug::lang::BinaryOp::Mul, xs2)?;
        self.c_proj.fwd(&xs)
    }
}

struct Attention<D: Device> {
    q_proj: Linear<D>,
    k_proj: Linear<D>,
    v_proj: Linear<D>,
    o_proj: Linear<D>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_interleaved: bool,
    custom_softmax: bool,
}

pub struct Cache<D: Device> {
    prev_k: LB<D>,
    prev_v: LB<D>,
}

impl<D: Device> Cache<D> {
    pub fn new(cfg: &Config, device: &D) -> Result<Vec<Self>> {
        let mut cache = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let prev_k = LB::cst(0f32, (1, cfg.num_key_value_heads, 0, cfg.head_dim()), device)?;
            let prev_v = LB::cst(0f32, (1, cfg.num_key_value_heads, 0, cfg.head_dim()), device)?;
            cache.push(Cache { prev_k, prev_v });
        }
        Ok(cache)
    }
}

impl<D: Device> Attention<D> {
    // We use a mutable cache rather than returning an updated value. This makes the function
    // signatures slightly simpler but introduces more mutability.
    fn fwd(&self, xs: &LB<D>, r: &Rope<D>, pos: &LB<D>, cache: &mut Cache<D>) -> Result<LB<D>> {
        let (_b_sz, seq_len, _hidden_size) = xs.shape().dims3()?;
        let q = self.q_proj.fwd(xs)?;
        let k = self.k_proj.fwd(xs)?;
        let v = self.v_proj.fwd(xs)?;

        let q = q.split_dim(2, self.num_heads, self.head_dim)?;
        let q = q.transpose(2, 1)?;
        let k = k.split_dim(2, self.num_kv_heads, self.head_dim)?;
        let k = k.transpose(2, 1)?;
        let v = v.split_dim(2, self.num_kv_heads, self.head_dim)?;
        let v = v.transpose(2, 1)?;

        let q = if self.rope_interleaved {
            D::rope_i(&q, &r.cos, &r.sin, pos)?
        } else {
            D::rope(&q, &r.cos, &r.sin, pos)?
        };
        let k = if self.rope_interleaved {
            D::rope_i(&k, &r.cos, &r.sin, pos)?
        } else {
            D::rope(&k, &r.cos, &r.sin, pos)?
        };
        let k = D::cat(&cache.prev_k, &k, 2)?;
        let v = D::cat(&cache.prev_v, &v, 2)?;

        cache.prev_k = k.clone();
        cache.prev_v = v.clone();
        let k = repeat(&k, 1, self.num_heads / self.num_kv_heads)?;
        let v = repeat(&v, 1, self.num_heads / self.num_kv_heads)?;

        // attention
        let att = q.matmul_t(k)?;
        let scale = LB::cst((self.head_dim as f32).powf(-0.5), (), q.device())?;
        let scale = scale.broadcast(att.shape())?;
        let att = att.binary(ug::lang::BinaryOp::Mul, scale)?;
        let att = if seq_len == 1 { att } else { D::causal_mask(&att)? };
        let att = if self.custom_softmax { D::custom_softmax(&att)? } else { softmax(&att)? };
        let xs = att.matmul(v)?;

        // final proj
        let xs = xs.transpose(2, 1)?;
        let xs = xs.merge_dims(ug::D::Minus2)?;
        let xs = self.o_proj.fwd(&xs)?;
        Ok(xs)
    }
}

struct Layer<D: Device> {
    rms1: RmsNorm<D>,
    attn: Attention<D>,
    rms2: RmsNorm<D>,
    mlp: Mlp<D>,
}

impl<D: Device> Layer<D> {
    fn fwd(&self, xs: &LB<D>, rope: &Rope<D>, pos: &LB<D>, cache: &mut Cache<D>) -> Result<LB<D>> {
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

struct Rope<D: Device> {
    cos: LB<D>,
    sin: LB<D>,
}

impl<D: Device> Rope<D> {
    fn new(cfg: &Config, dev: &D) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = LB::copy(theta.as_slice(), (1, head_dim / 2), dev)?;
        let idx_theta = LB::copy(
            (0..max_seq_len).map(|v| v as f32).collect::<Vec<_>>().as_slice(),
            (max_seq_len, 1),
            dev,
        )?;
        let mm = idx_theta.matmul(theta)?;
        let cos = mm.unary(ug::lang::UnaryOp::Cos)?;
        let sin = mm.unary(ug::lang::UnaryOp::Sin)?;
        Ok(Self { cos, sin })
    }
}

pub struct Model<D: Device> {
    embedding: LB<D>,
    rope: Rope<D>,
    layers: Vec<Layer<D>>,
    ln_f: RmsNorm<D>,
    lm_head: Linear<D>,
}

impl<D: Device> Model<D> {
    pub fn new(
        cfg: &Config,
        custom_softmax: bool,
        st: &ug::safetensors::MmapedSafetensors,
        dev: &D,
    ) -> Result<Self> {
        let embedding = st.load_with_cast("model.embed_tokens.weight", ug::DType::F32, dev)?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, "model.norm.weight", st, dev)?;
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
            let rms1 = RmsNorm::new(h_sz, eps, &format!("{name}.input_layernorm.weight"), st, dev)?;
            let rms2 = RmsNorm::new(
                h_sz,
                eps,
                &format!("{name}.post_attention_layernorm.weight"),
                st,
                dev,
            )?;
            let c_fc1 = Linear::new(h_sz, i_sz, &format!("{name}.mlp.gate_proj.weight"), st, dev)?;
            let c_fc2 = Linear::new(h_sz, i_sz, &format!("{name}.mlp.up_proj.weight"), st, dev)?;
            let c_proj = Linear::new(i_sz, h_sz, &format!("{name}.mlp.down_proj.weight"), st, dev)?;
            let q_proj =
                Linear::new(h_sz, h_sz, &format!("{name}.self_attn.q_proj.weight"), st, dev)?;
            let k_proj =
                Linear::new(h_sz, kv_sz, &format!("{name}.self_attn.k_proj.weight"), st, dev)?;
            let v_proj =
                Linear::new(h_sz, kv_sz, &format!("{name}.self_attn.v_proj.weight"), st, dev)?;
            let o_proj =
                Linear::new(h_sz, h_sz, &format!("{name}.self_attn.o_proj.weight"), st, dev)?;
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
        let rope = Rope::new(cfg, dev)?;
        Ok(Self { embedding, layers, ln_f, lm_head, rope })
    }

    pub fn fwd(&self, tokens: &LB<D>, pos: &LB<D>, cache: &mut [Cache<D>]) -> Result<LB<D>> {
        let mut xs = index_select(&self.embedding, tokens)?;
        for (layer, cache) in self.layers.iter().zip(cache) {
            xs = layer.fwd(&xs, &self.rope, pos, cache)?
        }
        let xs = self.ln_f.fwd(&xs)?;
        let xs = self.lm_head.fwd(&xs)?;
        Ok(xs)
    }
}
