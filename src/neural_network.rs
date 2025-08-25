use crate::{loss::LossFunction, Layer, NeuralNetwork};

impl<'a> NeuralNetwork<'a> {
    pub fn new(layers: &'a mut [Layer]) -> Self {
        let n: usize = layers.len();
        assert!(n > 0, "NeuralNetwork must have at least one layer.");

        NeuralNetwork {
            input_size: layers[0].input_size,
            output_size: layers[n - 1].output_size,
            layers,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_size, "Input size mismatch.");

        let n = self.layers.len();
        let mut buffer1 = input.to_vec();
        let mut buffer2 = vec![0.0; self.output_size];

        let mut input_buffer = &mut buffer1;
        let mut output_buffer = &mut buffer2;

        for (i, layer) in self.layers.iter().enumerate() {
            if i < n - 1 {
                output_buffer.resize(layer.output_size, 0.0);
            }

            *output_buffer = layer.forward(input_buffer);
            std::mem::swap(&mut input_buffer, &mut output_buffer);
        }

        input_buffer.clone()
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        learning_rate: f32,
        loss_function: LossFunction,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let mut activations: Vec<Vec<f32>> = Vec::new();

                let mut temp_input = input.clone();
                activations.push(temp_input.clone());

                for layer in &mut *self.layers {
                    temp_input = layer.forward(&temp_input);
                    activations.push(temp_input.clone());
                }

                total_loss += (loss_function.function)(activations.last().unwrap(), target);

                let mut grad = (loss_function.derivative)(activations.last().unwrap(), target);

                for i in (0..self.layers.len()).rev() {
                    grad = self.layers[i].backward(&activations[i], &grad, learning_rate);
                }
            }

            let average_loss = total_loss / inputs.len() as f32;
            println!("Epoch {}: Loss = {}", epoch + 1, average_loss);
        }
    }
}
