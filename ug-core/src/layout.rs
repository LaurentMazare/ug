//! The shape of a tensor is a tuple with the size of each of its dimensions.
use crate::{bail, Error, Result};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape(Vec<usize>);

pub const SCALAR: Shape = Shape(vec![]);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

impl From<(usize,)> for Shape {
    fn from(d1: (usize,)) -> Self {
        Self(vec![d1.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d12: (usize, usize)) -> Self {
        Self(vec![d12.0, d12.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d123: (usize, usize, usize)) -> Self {
        Self(vec![d123.0, d123.1, d123.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d1234: (usize, usize, usize, usize)) -> Self {
        Self(vec![d1234.0, d1234.1, d1234.2, d1234.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(d12345: (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![d12345.0, d12345.1, d12345.2, d12345.3, d12345.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(d123456: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self(vec![d123456.0, d123456.1, d123456.2, d123456.3, d123456.4, d123456.5])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(dims: &[usize]) -> Result<$out_type> {
            if dims.len() != $cnt {
                bail!(
                    "unexpected number of dims, expected {} got {} shape {:?}",
                    $cnt,
                    dims.len(),
                    dims
                )
            }
            Ok($dims(dims))
        }

        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                $fn_name(self.0.as_slice())
            }
        }

        impl std::convert::TryInto<$out_type> for Shape {
            type Error = crate::Error;
            fn try_into(self) -> std::result::Result<$out_type, Self::Error> {
                self.$fn_name()
            }
        }
    };
}

impl Shape {
    pub fn num_elements(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    /// The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    /// The dimensions as a slice of `usize`.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// The total number of elements, this is the product of all dimension sizes.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// The strides given in number of elements for a contiguous n-dimensional
    /// arrays using this shape.
    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    /// Returns true if the strides are C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    /// Returns true if the strides are Fortran contiguous (aka column major).
    pub fn is_fortran_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()) {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    /// Modifies the shape by adding a list of additional dimensions at the end of the existing
    /// dimensions.
    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }

    /// Check whether the two shapes are compatible for broadcast, and if it is the case return the
    /// broadcasted shape. This is to be used for binary pointwise ops.
    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx { 1 } else { lhs_dims[lhs_ndims - rev_idx] };
            let r_value = if rhs_ndims < rev_idx { 1 } else { rhs_dims[rhs_ndims - rev_idx] };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                bail!("shape mismatch in binary op '{op}', lhs: {lhs:?} rhs: {rhs:?}")
            }
        }
        Ok(Shape::from(bcast_dims))
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            bail!("dim out of range in '{op}', dim: {dim}, shape: {shape:?}")
        }
        Ok(dim)
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.dims().len() {
            bail!("dim out of range in '{op}', dim: {dim}, shape: {shape:?}")
        }
        Ok(dim)
    }
}

impl Dim for i32 {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= 0 {
            (dim as usize).to_index(shape, op)
        } else {
            D::Minus((-dim) as usize).to_index(shape, op)
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= 0 {
            (dim as usize).to_index_plus_one(shape, op)
        } else {
            D::Minus((-dim) as usize).to_index_plus_one(shape, op)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
        };
        Error::Msg(format!("dim out of range in '{op}', dim: {dim}, shape: {shape:?}")).bt()
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

pub trait Dims: Sized {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;

    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dims = self.to_indexes_internal(shape, op)?;
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                bail!("duplicate dim indexes in '{op}', dims: {dims:?}, shape: {shape:?}")
            }
            if dim >= shape.rank() {
                bail!("dim out of range in '{op}', dim: {dim}, shape: {shape:?}")
            }
        }
        Ok(dims)
    }
}

impl Dims for Vec<usize> {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for &[usize] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for () {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(vec![])
    }
}

impl<D: Dim + Sized> Dims for D {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D: Dim> Dims for (D,) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.0.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok(vec![d0, d1])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok(vec![d0, d1, d2])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3, d4])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> Dims for (D1, D2, D3, D4, D5, D6) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        let d5 = self.5.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3, d4, d5])
    }
}

extract_dims!(dims0, 0, |_: &[usize]| (), ());
extract_dims!(dims1, 1, |d: &[usize]| d[0], usize);
extract_dims!(dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(dims3, 3, |d: &[usize]| (d[0], d[1], d[2]), (usize, usize, usize));
extract_dims!(dims4, 4, |d: &[usize]| (d[0], d[1], d[2], d[3]), (usize, usize, usize, usize));
extract_dims!(
    dims5,
    5,
    |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]),
    (usize, usize, usize, usize, usize)
);

pub trait ShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape>;
}

impl<S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape> {
        Ok(self.into())
    }
}

impl ShapeWithOneHole for ((),) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        Ok(el_count.into())
    }
}

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if el_count % prod_d != 0 {
        bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for ((), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1) = self;
        Ok((hole_size(el_count, d1, &self)?, d1).into())
    }
}

impl ShapeWithOneHole for (usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, ()) = self;
        Ok((d1, hole_size(el_count, d1, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2) = self;
        Ok((hole_size(el_count, d1 * d2, &self)?, d1, d2).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2) = self;
        Ok((d1, hole_size(el_count, d1 * d2, &self)?, d2).into())
    }
}

impl ShapeWithOneHole for (usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, ()) = self;
        Ok((d1, d2, hole_size(el_count, d1 * d2, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d, d1, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d3, d).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d, d1, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, (), d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d4, d).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous(), Vec::<usize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous(), [1]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous(), [1337, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous(), [458 * 792, 458, 1]);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    // TODO: We only have a single offset here, maybe we should have one offset per dimension?
    offset: usize,
}

impl Layout {
    pub fn can_be_compressed(&self) -> bool {
        let strides = self.strides();
        let dims = self.dims();
        if dims.len() <= 1 {
            return false;
        }
        for i in 0..dims.len() - 1 {
            if strides[i] != strides[i + 1] * dims[i + 1] {
                return false;
            }
        }
        true
    }

    pub fn compress_all(&self) -> Result<Self> {
        let strides = self.strides();
        let dims = self.dims();
        for i in 0..dims.len() - 1 {
            if strides[i] != strides[i + 1] * dims[i + 1] {
                bail!("cannot collapse dims, {self:?}")
            }
        }
        let stride = strides.last().copied().unwrap_or(1);
        let dim = self.num_elements();
        Ok(Self { shape: Shape::from(dim), strides: vec![stride], offset: self.offset() })
    }

    pub fn from_shape<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        let mut strides = vec![];
        let mut stride = 1;
        for l in shape.dims().iter().rev() {
            strides.push(stride);
            stride *= l
        }
        strides.reverse();
        Self { shape, strides, offset: 0 }
    }

    pub fn transpose(&self) -> Self {
        let r = self.rank();
        if r < 2 {
            return self.clone();
        }
        let mut dims = self.dims().to_vec();
        let mut strides = self.strides.to_vec();
        dims.swap(r - 2, r - 1);
        strides.swap(r - 2, r - 1);
        Self { shape: dims.into(), offset: self.offset, strides }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset
    }

    pub fn c_contiguous(&self) -> bool {
        let mut prod_l = 1;
        for (&s, &l) in self.strides.iter().zip(self.shape.dims().iter()).rev() {
            if s != prod_l {
                return false;
            }
            prod_l *= l
        }
        true
    }
}
