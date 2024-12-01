use ug::Result;

#[test]
fn lb_cuda_add() -> Result<()> {
    let dev = ug_metal::Device::new()?;
    ug::common_tests::lb::add(&dev)
}

#[test]
fn lb_cuda_mm() -> Result<()> {
    let dev = ug_metal::Device::new()?;
    ug::common_tests::lb::mm(&dev)
}
