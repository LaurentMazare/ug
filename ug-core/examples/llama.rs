// wget https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
use ug::Result;

fn main() -> Result<()> {
    let _st = unsafe { ug::safetensors::MmapedSafetensors::new("model.safetensors")? };
    Ok(())
}
