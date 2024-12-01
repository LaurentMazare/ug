use crate::{bail, Device, LazyBuffer as LB, Result};

pub mod lb {
    use super::*;

    pub fn add<D: Device>(dev: &D) -> Result<()> {
        let lhs = LB::cst(40., (5, 2), dev)?;
        let rhs = LB::cst(2., (5, 2), dev)?;
        let lb = lhs.binary(crate::lang::BinaryOp::Add, rhs)?;
        lb.realize()?;
        let data = match lb.data_vec::<f32>()? {
            None => bail!("no data"),
            Some(v) => v,
        };
        assert_eq!(data, [42., 42., 42., 42., 42., 42., 42., 42., 42., 42.]);
        Ok(())
    }

    pub fn mm<D: Device>(dev: &D) -> Result<()> {
        let lhs = LB::cst(1., (4, 2), dev)?;
        let rhs = LB::cst(2., (2, 3), dev)?;
        let lb = lhs.matmul(rhs)?;
        lb.realize()?;
        let data = match lb.data_vec::<f32>()? {
            None => bail!("no data"),
            Some(v) => v,
        };
        assert_eq!(data, [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
        Ok(())
    }
}
