use std::sync::Arc;
extern crate test;

#[derive(Debug, Clone)]
pub struct Tensor<T, const SIZE: usize, const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
    data: Arc<[T]>,
}

pub trait TensorOps<T> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, idx: &[usize]) -> Option<&T>;
    fn set(&mut self, idx: &[usize], value: T) -> Result<(), &'static str>;

    fn add(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Add<Output = T>,
        Self: Sized;
    fn sub(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Sub<Output = T>,
        Self: Sized;
    fn mul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T>,
        Self: Sized;
    fn div(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Div<Output = T>,
        Self: Sized;

    fn matmul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default,
        Self: Sized;
}

pub trait TensorMathOps {
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
}

#[macro_export]
macro_rules! elementwise_op {
    ($self:expr, $rhs:expr, $op:tt) => {{
        let mut iter = $rhs.data.iter();
        Tensor {
            shape: $rhs.shape,
            strides: $rhs.strides,
            data: $self.data.iter().map(|x| *x $op *iter.next().unwrap()).collect(),
        }
    }};
}

#[macro_export]
macro_rules! impl_elementwise {
    ($op:ident, $trait_op:ident, $op_fn:ident) => {
        impl<T, const SIZE: usize, const N: usize> std::ops::$trait_op<Tensor<T, SIZE, N>>
            for Tensor<T, SIZE, N>
        where
            T: Default + Copy + std::ops::$trait_op<Output = T> + Sync + Send,
            rand::distributions::Standard: rand::distributions::Distribution<T>,
        {
            type Output = Tensor<T, SIZE, N>;

            fn $op_fn(self, rhs: Tensor<T, SIZE, N>) -> Self::Output {
                (&self).$op(&rhs)
            }
        }
    };
}

impl<T: Default + Copy, const SIZE: usize, const N: usize> Tensor<T, SIZE, N>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    pub fn new(shape: [usize; N]) -> Self {
        for &dim in &shape {
            assert!(dim >= 1, "Dimension must be >= 1 but got {}", dim);
        }
        let size: usize = shape.iter().product();
        let data = Arc::from(vec![T::default(); size]);
        let mut strides = [0; N];
        let mut acc = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            strides[i] = acc;
            acc *= shape[i];
        }
        Tensor {
            data,
            shape,
            strides,
        }
    }

    pub fn rand(shape: [usize; N]) -> Self {
        for &dim in &shape {
            assert!(dim >= 1, "Dimension must be >= 1 but got {}", dim);
        }
        let size: usize = shape.iter().product();
        let data = (0..size)
            .map(|_| rand::random::<T>())
            .collect::<Vec<T>>()
            .into();
        let mut strides = [0; N];
        let mut acc = 1;
        let mut i = N;
        while i > 0 {
            i -= 1;
            strides[i] = acc;
            acc *= shape[i];
        }
        Tensor {
            data,
            shape,
            strides,
        }
    }

    pub fn set_data(&mut self, data: &[T]) {
        assert!(
            data.len() == self.data.len(),
            "Input data length does not match tensor size"
        );
        let slice = Arc::make_mut(&mut self.data);
        slice.copy_from_slice(data);
    }

    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize; N] {
        &self.strides
    }

    pub fn get(&self, idx: [usize; N]) -> Option<&T> {
        let mut offset = 0;
        for (d, &i) in idx.iter().enumerate() {
            if i >= self.shape[d] {
                return None;
            }
            offset += i * self.strides[d];
        }
        Some(&self.data[offset])
    }
}

impl<T: Default + Copy, const SIZE: usize, const N: usize> TensorOps<T> for Tensor<T, SIZE, N>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn get(&self, idx: &[usize]) -> Option<&T> {
        if idx.len() != N {
            return None;
        }
        let idx_array: [usize; N] = idx.try_into().ok()?;
        self.get(idx_array)
    }

    fn set(&mut self, idx: &[usize], value: T) -> Result<(), &'static str> {
        if idx.len() != N {
            return Err("Invalid index dimension");
        }
        let mut offset = 0;
        for (d, &i) in idx.iter().enumerate() {
            if i >= self.shape[d] {
                return Err("Index out of bounds");
            }
            offset += i * self.strides[d];
        }
        let slice = Arc::make_mut(&mut self.data);
        slice[offset] = value;
        Ok(())
    }

    fn add(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Add<Output = T>,
    {
        elementwise_op!(self, rhs, +)
    }

    fn sub(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Sub<Output = T>,
    {
        elementwise_op!(self, rhs, -)
    }

    fn mul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        elementwise_op!(self, rhs, *)
    }

    fn div(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Div<Output = T>,
    {
        elementwise_op!(self, rhs, /)
    }

    fn matmul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default,
    {
        assert!(
            self.shape.len() >= 2 && rhs.shape.len() >= 2,
            "Tensors must have rank >= 2"
        );
        assert_eq!(self.shape.len(), rhs.shape.len(), "Tensor ranks must match");

        let dim = self.shape.len();
        let m = self.shape[dim - 2];
        let k = self.shape[dim - 1];
        let k2 = rhs.shape[dim - 2];
        let n = rhs.shape[dim - 1];
        let batch_shape = &self.shape[..dim - 2];

        assert_eq!(
            batch_shape,
            &rhs.shape[..dim - 2],
            "Batch dimensions must match"
        );
        assert_eq!(k, k2, "Inner matrix dimensions must match");

        let mut result_shape = [0; N];
        result_shape[..(dim - 2)].copy_from_slice(&self.shape[..(dim - 2)]);
        result_shape[dim - 2] = m;
        result_shape[dim - 1] = n;

        let mut result = Self::new(result_shape);

        let batch_size: usize = batch_shape.iter().product();

        for batch in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::default();
                    for kk in 0..k {
                        let a_index = batch * m * k + i * k + kk;
                        let b_index = batch * k * n + kk * n + j;
                        sum = sum + self.data[a_index] * rhs.data[b_index];
                    }
                    let result_index = batch * m * n + i * n + j;
                    Arc::make_mut(&mut result.data)[result_index] = sum;
                }
            }
        }

        result
    }
}

impl_elementwise!(add, Add, add);
impl_elementwise!(sub, Sub, sub);
impl_elementwise!(mul, Mul, mul);
impl_elementwise!(div, Div, div);

impl<T: Copy + Into<f64> + From<f64>, const SIZE: usize, const N: usize> TensorMathOps
    for Tensor<T, SIZE, N>
where
    T: Copy,
{
    fn exp(&self) -> Self {
        let data = self.data.iter().map(|&x| T::from(x.into().exp())).collect();
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data,
        }
    }

    fn ln(&self) -> Self {
        let data = self.data.iter().map(|&x| T::from(x.into().ln())).collect();
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data,
        }
    }

    fn sin(&self) -> Self {
        let data = self.data.iter().map(|&x| T::from(x.into().sin())).collect();
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data,
        }
    }

    fn cos(&self) -> Self {
        let data = self.data.iter().map(|&x| T::from(x.into().cos())).collect();
        Tensor {
            shape: self.shape,
            strides: self.strides,
            data,
        }
    }
}

#[macro_export]
macro_rules! tensor {
    ($t:ty, $($dim:expr),+) => {{
        const SIZE: usize = product!($($dim),+);
        const DIMS: usize = count_exprs!($($dim),+);
        Tensor::<$t, SIZE, DIMS>::new([$($dim),+])
    }};
}

#[macro_export]
macro_rules! tensor_rand {
    ($t:ty, $($dim:expr),+) => {{
        const SIZE: usize = product!($($dim),+);
        const DIMS: usize = count_exprs!($($dim),+);
        Tensor::<$t, SIZE, DIMS>::rand([$($dim),+])
    }};
}

#[allow(unused)]
macro_rules! product {
    ($x:expr) => { $x };
    ($x:expr, $($xs:expr),+) => { $x * product!($($xs),+) };
}

#[allow(unused)]
macro_rules! count_exprs {
    ($($x:expr),*) => {
        <[()]>::len(&[$(count_exprs!(@sub $x)),*])
    };
    (@sub $x:expr) => { () };
}

#[cfg(test)]
mod tests {
    use super::test::Bencher;
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t1 = tensor!(f32, 5);
        assert_eq!(t1.shape(), &[5]);
        assert_eq!(t1.strides(), &[1]);

        let t2 = tensor!(f32, 2, 5);
        assert_eq!(t2.shape(), &[2, 5]);
        assert_eq!(t2.strides(), &[5, 1]);

        let t3 = tensor!(f32, 2, 3, 4);
        assert_eq!(t3.shape(), &[2, 3, 4]);
        assert_eq!(t3.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_ui() {
        let mut t = tensor!(f32, 5);
        t.set_data(&[0.0, 1.0, 2.0, 3.0, 4.0]);

        assert_eq!(t.get([3]), Some(&3.0));
        assert_eq!(t.get([5]), None);

        t.set(&[2], 10.0).unwrap();
        assert_eq!(t.get([2]), Some(&10.0));

        assert!(t.set(&[5], 1.0).is_err());
        assert!(t.set(&[0, 0], 1.0).is_err());
    }

    #[test]
    fn test_elementwise_ops() {
        let mut a = tensor!(f32, 2, 2);
        let mut b = tensor!(f32, 2, 2);

        a.set_data(&[1.0, 2.0, 3.0, 4.0]);
        b.set_data(&[1.0, 1.0, 1.0, 1.0]);

        let c = a.add(&b);
        let d = a.sub(&b);
        let e = a.mul(&b);
        let f = a.div(&b);

        assert_eq!(c.get([0, 0]), Some(&2.0));
        assert_eq!(d.get([0, 0]), Some(&0.0));
        assert_eq!(e.get([0, 0]), Some(&1.0));
        assert_eq!(f.get([0, 0]), Some(&1.0));
    }

    #[test]
    fn test_tensor_matmul() {
        let mut a = tensor!(f32, 2, 3);
        let mut b = tensor!(f32, 3, 2);

        a.set_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        b.set_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let c = a.matmul(&b);

        assert_eq!(c.get([0, 0]), Some(&22.0));
        assert_eq!(c.get([0, 1]), Some(&28.0));
        assert_eq!(c.get([1, 0]), Some(&49.0));
        assert_eq!(c.get([1, 1]), Some(&64.0));
    }

    #[bench]
    fn bench_add(b: &mut Bencher) {
        let t1 = tensor!(f32, 100, 100);
        let t2 = tensor!(f32, 100, 100);
        b.iter(|| t1.clone() + t2.clone())
    }

    #[bench]
    fn bench_matmul(b: &mut Bencher) {
        let t1 = tensor!(f32, 100, 100);
        let t2 = tensor!(f32, 100, 100);
        b.iter(|| t1.matmul(&t2))
    }

    #[bench]
    fn bench_candle_add(b: &mut Bencher) {
        let device = candle_core::Device::Cpu;
        let t1 = candle_core::Tensor::randn(0f32, 1., (100, 100), &device).unwrap();
        let t2 = candle_core::Tensor::randn(0f32, 1., (100, 100), &device).unwrap();
        b.iter(|| t1.clone() + t2.clone())
    }

    #[bench]
    fn bench_candle_matmul(b: &mut Bencher) {
        let device = candle_core::Device::Cpu;
        let t1 = candle_core::Tensor::randn(0f32, 1., (100, 100), &device).unwrap();
        let t2 = candle_core::Tensor::randn(0f32, 1., (100, 100), &device).unwrap();
        b.iter(|| t1.matmul(&t2))
    }

    #[bench]
    fn bench_large_add(b: &mut Bencher) {
        let t1 = tensor!(f32, 1000, 1000);
        let t2 = tensor!(f32, 1000, 1000);
        b.iter(|| t1.clone() + t2.clone())
    }

    #[bench]
    fn bench_candle_large_add(b: &mut Bencher) {
        let device = candle_core::Device::Cpu;
        let t1 = candle_core::Tensor::randn(0f32, 1., (1000, 1000), &device).unwrap();
        let t2 = candle_core::Tensor::randn(0f32, 1., (1000, 1000), &device).unwrap();
        b.iter(|| t1.clone() + t2.clone())
    }

    #[bench]
    fn bench_small_batched_matmul(b: &mut Bencher) {
        let t1 = tensor!(f32, 10, 32, 32);
        let t2 = tensor!(f32, 10, 32, 32);
        b.iter(|| t1.matmul(&t2))
    }

    #[bench]
    fn bench_candle_small_batched_matmul(b: &mut Bencher) {
        let device = candle_core::Device::Cpu;
        let t1 = candle_core::Tensor::randn(0f32, 1., (10, 32, 32), &device).unwrap();
        let t2 = candle_core::Tensor::randn(0f32, 1., (10, 32, 32), &device).unwrap();
        b.iter(|| t1.matmul(&t2))
    }

    use super::*;

    #[test]
    fn test_exp() {
        let mut t = tensor!(f64, 3);
        t.set_data(&[0.0, 1.0, 2.0]);
        let res = t.exp();
        assert!((res.get([0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((res.get([1]).unwrap() - std::f64::consts::E).abs() < 1e-10);
        assert!((res.get([2]).unwrap() - std::f64::consts::E.powi(2)).abs() < 1e-10);
    }

    #[test]
    fn test_ln() {
        let mut t = tensor!(f64, 3);
        t.set_data(&[1.0, std::f64::consts::E, std::f64::consts::E.powi(2)]);
        let res = t.ln();
        assert!((res.get([0]).unwrap() - 0.0).abs() < 1e-10);
        assert!((res.get([1]).unwrap() - 1.0).abs() < 1e-10);
        assert!((res.get([2]).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_cos() {
        let mut t = tensor!(f64, 2);
        t.set_data(&[0.0, std::f64::consts::FRAC_PI_2]);
        let sin_res = t.sin();
        let cos_res = t.cos();
        assert!((sin_res.get([0]).unwrap() - 0.0).abs() < 1e-10);
        assert!((sin_res.get([1]).unwrap() - 1.0).abs() < 1e-10);
        assert!((cos_res.get([0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((cos_res.get([1]).unwrap() - 0.0).abs() < 1e-10);
    }
}
