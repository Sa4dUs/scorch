pub struct LossFunction {
    pub function: fn(&[f32], &[f32]) -> f32,
    pub derivative: fn(&[f32], &[f32]) -> Vec<f32>,
}

pub const MSE: LossFunction = LossFunction {
    function: mean_squared_error,
    derivative: mse_derivative,
};

fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
}

fn mse_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| 2.0 * (pred - target))
        .collect()
}

pub const MAE: LossFunction = LossFunction {
    function: mean_absolute_error,
    derivative: mae_derivative,
};

fn mean_absolute_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).abs())
        .sum::<f32>()
        / predictions.len() as f32
}

fn mae_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| if pred > target { 1.0 } else { -1.0 })
        .collect()
}

pub const CROSS_ENTROPY: LossFunction = LossFunction {
    function: cross_entropy,
    derivative: cross_entropy_derivative,
};

fn cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            if *target == 1.0 {
                -target * pred.ln()
            } else {
                -(1.0 - target) * (1.0 - pred).ln()
            }
        })
        .sum::<f32>()
}

fn cross_entropy_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            if *target == 1.0 {
                -1.0 / pred
            } else {
                1.0 / (1.0 - pred)
            }
        })
        .collect()
}
