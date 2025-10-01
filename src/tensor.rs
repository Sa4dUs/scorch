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
    fn tensor_mul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default,
        Self: Sized;
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
            return Err("Invalid index dimensions");
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

    fn tensor_mul(&self, rhs: &Self) -> Self
    where
        T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default,
    {
        assert!(
            self.shape.len() >= 2 && rhs.shape.len() >= 2,
            "Tensors must have at least 2 dimensions"
        );
        assert_eq!(
            self.shape[N - 1],
            rhs.shape[N - 2],
            "Inner dimensions must match"
        );

        let batch_dims = &self.shape[..N - 2];
        let m = self.shape[N - 2];
        let k = self.shape[N - 1];
        let n = rhs.shape[N - 1];

        let mut result = Self::new(self.shape);
        let batch_size: usize = batch_dims.iter().product();

        for batch in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::default();
                    for l in 0..k {
                        let a_idx = batch * (m * k) + i * k + l;
                        let b_idx = batch * (k * n) + l * n + j;
                        sum = sum + self.data[a_idx] * rhs.data[b_idx];
                    }
                    let result_idx = batch * (m * n) + i * n + j;
                    Arc::make_mut(&mut result.data)[result_idx] = sum;
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

macro_rules! product {
    ($x:expr) => { $x };
    ($x:expr, $($xs:expr),+) => { $x * product!($($xs),+) };
}

macro_rules! count_exprs {
    ($($x:expr),*) => { <[()]>::len(&[$(count_exprs!(@sub $x)),*]) };
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

        let c = a.tensor_mul(&b);

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
        b.iter(|| t1.tensor_mul(&t2))
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
        b.iter(|| t1.tensor_mul(&t2))
    }

    #[bench]
    fn bench_candle_small_batched_matmul(b: &mut Bencher) {
        let device = candle_core::Device::Cpu;
        let t1 = candle_core::Tensor::randn(0f32, 1., (10, 32, 32), &device).unwrap();
        let t2 = candle_core::Tensor::randn(0f32, 1., (10, 32, 32), &device).unwrap();
        b.iter(|| t1.matmul(&t2))
    }
}
