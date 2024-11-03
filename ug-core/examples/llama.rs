// wget https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
use ug::{Result, Slice};

fn main() -> Result<()> {
    let dev = &ug::CpuDevice;
    let st = unsafe { ug::safetensors::MmapedSafetensors::new("model.safetensors")? };
    let tensor = st.load("model.embed_tokens.weight", dev)?;
    println!("{:?} {:?} {}", tensor.shape(), tensor.dtype(), tensor.realized());
    {
        let data = tensor.data().lock().unwrap();
        let data = data.as_ref().unwrap();
        let data = data.to_vec::<half::bf16>()?;
        println!("{:?}", &data[..10]);
    };
    let tensor = tensor.cast(ug::DType::F32)?;
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
