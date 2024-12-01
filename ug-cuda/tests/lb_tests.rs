use ug::Result;

#[test]
fn lb_cuda_add() -> Result<()> {
    let dev = ug_cuda::Device::new(0)?;
    ug::common_tests::lb::add(&dev)
}

#[test]
fn lb_cuda_mm() -> Result<()> {
    let dev = ug_cuda::Device::new(0)?;
    ug::common_tests::lb::mm(&dev)
}

#[test]
fn lb_cuda_cat() -> Result<()> {
    ug::common_tests::lb::cat(&ug::CpuDevice)
}
