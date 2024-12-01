use ug::Result;

#[test]
fn lb_metal_add() -> Result<()> {
    let dev = ug_metal::Device::new()?;
    ug::common_tests::lb::add(&dev)
}

#[test]
fn lb_metal_mm() -> Result<()> {
    let dev = ug_metal::Device::new()?;
    ug::common_tests::lb::mm(&dev)
}

#[test]
fn lb_metal_cat() -> Result<()> {
    ug::common_tests::lb::cat(&ug::CpuDevice)
}
