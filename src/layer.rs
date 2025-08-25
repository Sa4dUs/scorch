use crate::{Layer, activation::ActivationFunction};
use rand::Rng;

impl Layer {
    pub fn new(in_size: usize, out_size: usize, act_f: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_size as f32).sqrt();

        let weights: Vec<f32> = (0..in_size * out_size)
            .map(|_| ((rng.r#gen::<f32>() - 0.5) * 2.0) * scale)
            .collect();
        let biases: Vec<f32> = vec![0.0; out_size];

        Layer {
            weights,
            biases,
            input_size: in_size,
            output_size: out_size,
            activation_function: act_f,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];

        (0..self.output_size).for_each(|i| {
            output[i] = self.biases[i];

            (0..self.input_size).for_each(|j| {
                output[i] += input[j] * self.weights[j * self.output_size + i];
            });
        });

        (self.activation_function)(&output)
    }

    pub fn backward(&mut self, input: &[f32], output_grad: &[f32], lr: f32) -> Vec<f32> {
        let mut input_grad = vec![0.0; self.input_size];

        (0..self.output_size).for_each(|i| {
            for j in 0..self.input_size {
                let idx = j * self.output_size + i;
                let grad = output_grad[i] * input[j];
                self.weights[idx] -= lr * grad;
                input_grad[j] += output_grad[i] * self.weights[idx];
            }
            self.biases[i] -= lr * output_grad[i];
        });

        input_grad
    }
}
