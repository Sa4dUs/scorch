pub type ActivationFunction = fn(&[f32]) -> Vec<f32>;

pub fn sigmoid(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0 / (1.0 + f32::exp(-x))).collect()
}

pub fn sigmoid_derivative(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let sig = 1.0 / (1.0 + f32::exp(-x));
            sig * (1.0 - sig)
        })
        .collect()
}

pub fn tanh(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| f32::tanh(x)).collect()
}

pub fn tanh_derivative(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            let tanh_val = f32::tanh(x);
            1.0 - tanh_val.powi(2)
        })
        .collect()
}

pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = input.iter().map(|&x| f32::exp(x - max)).collect();
    let sum: f32 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum).collect()
}

pub fn relu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect()
}

pub fn linear(input: &[f32]) -> Vec<f32> {
    input.to_vec()
}
