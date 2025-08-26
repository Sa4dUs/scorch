#[derive(Debug)]
#[repr(transparent)]
struct Tensor<T> {
    data: T,
}

impl<T> Tensor<T>
where
    T: Default,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn zero() -> Self {
        Tensor { data: T::default() }
    }

    fn rand() -> Self {
        Tensor {
            data: rand::random(),
        }
    }
}

#[macro_export]
macro_rules! tensor {
    ($t:ty, $($args:expr),+) => {
        Tensor<ndarray!($t, $($args),+)>
    }
}

macro_rules! ndarray {
    ($t:ty, $n: expr) => {
        [$t; $n]
    };
    ($t:ty, $n: expr, $($args:expr),+) => {
        [ndarray!($t, $($args),+); $n]
    }
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use crate::tensor::Tensor;

    #[test]
    fn dim_1() {
        assert_eq!(
            TypeId::of::<tensor!(f32, 5)>(),
            TypeId::of::<Tensor<[f32; 5]>>()
        );
    }

    #[test]
    fn dim_2() {
        assert_eq!(
            TypeId::of::<tensor!(f32, 2, 5)>(),
            TypeId::of::<Tensor<[[f32; 5]; 2]>>()
        );
    }

    #[test]
    fn dim_3() {
        assert_eq!(
            TypeId::of::<tensor!(f32, 6, 7, 2)>(),
            TypeId::of::<Tensor<[[[f32; 2]; 7]; 6]>>()
        );
    }

    #[test]
    fn dim_n() {
        assert_eq!(
            TypeId::of::<tensor!(f32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)>(),
            TypeId::of::<Tensor<[[[[[[[[[[f32; 10]; 9]; 8]; 7]; 6]; 5]; 4]; 3]; 2]; 1]>>()
        );
    }

    #[test]
    fn ui() {
        let t_zero = <tensor!(f32, 5)>::zero();
        dbg!(t_zero);

        let t_rand = <tensor!(f32, 6, 6)>::rand();
        dbg!(t_rand);
    }
}
