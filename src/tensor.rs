#[derive(Debug)]
pub struct Tensor<T, const SIZE: usize, const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
    data: Box<[T]>,
}

impl<T: Default + Copy, const SIZE: usize, const N: usize> Tensor<T, SIZE, N>
where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    pub fn new(shape: [usize; N]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::default(); size].into_boxed_slice();
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
        let data = (0..size).map(|_| rand::random::<T>()).collect::<Box<[T]>>();
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
        self.data.copy_from_slice(data);
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

impl<T, const SIZE: usize, const N: usize> std::ops::Add<Tensor<T, SIZE, N>> for Tensor<T, SIZE, N>
where
    T: Default + Copy + std::ops::Add<Output = T>,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    type Output = Tensor<T, SIZE, N>;

    fn add(self, rhs: Tensor<T, SIZE, N>) -> Self::Output {
        let mut out = Tensor::new(rhs.shape);
        let data = self
            .data
            .iter()
            .map(|x| *x + *rhs.data.iter().next().unwrap())
            .collect::<Vec<_>>();
        out.set_data(&data);
        out
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
    use crate::tensor::Tensor;

    #[test]
    fn dim_1() {
        let t = tensor!(f32, 5);
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.strides(), &[1]);
        assert_eq!(t.data.len(), 5);
    }

    #[test]
    fn dim_2() {
        let t = tensor!(f32, 2, 5);
        assert_eq!(t.shape(), &[2, 5]);
        assert_eq!(t.strides(), &[5, 1]);
        assert_eq!(t.data.len(), 10);
    }

    #[test]
    fn dim_3() {
        let t = tensor!(f32, 6, 7, 2);
        assert_eq!(t.shape(), &[6, 7, 2]);
        assert_eq!(t.strides(), &[14, 2, 1]);
        assert_eq!(t.data.len(), 84);
    }

    #[test]
    fn dim_n() {
        let t = tensor!(u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        assert_eq!(t.shape(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(t.data.len(), 3628800);
    }

    #[test]
    fn ui() {
        let mut t = tensor!(f32, 5);
        t.data.copy_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);

        assert_eq!(t.get([3]), Some(&3.0));
        assert_eq!(t.get([5]), None);
    }

    #[test]
    fn sum() {
        let a = tensor_rand!(f32, 10);
        let b = tensor_rand!(f32, 10);
        let c = a + b;
        dbg!(c);
    }
}
