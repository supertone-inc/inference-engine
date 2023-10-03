use thiserror::Error;

pub trait InferenceEngine {
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;

    fn input_shape(&self, index: usize) -> &[usize];
    fn input_shapes(&self) -> Vec<&[usize]> {
        (0..self.input_count())
            .map(|i| self.input_shape(i))
            .collect()
    }
    fn output_shape(&self, index: usize) -> &[usize];
    fn output_shapes(&self) -> Vec<&[usize]> {
        (0..self.output_count())
            .map(|i| self.output_shape(i))
            .collect()
    }

    fn set_input_shape(&mut self, index: usize, shape: &[usize]) -> Result<(), Error>;
    fn set_input_shapes(&mut self, shapes: &[&[usize]]) -> Result<(), Error> {
        shapes
            .iter()
            .enumerate()
            .try_for_each(|(i, shape)| self.set_input_shape(i, shape))
    }
    fn set_output_shape(&mut self, index: usize, shape: &[usize]) -> Result<(), Error>;
    fn set_output_shapes(&mut self, shapes: &[&[usize]]) -> Result<(), Error> {
        shapes
            .iter()
            .enumerate()
            .try_for_each(|(i, shape)| self.set_output_shape(i, shape))
    }

    fn input_data(&mut self, index: usize) -> &mut [f32];
    fn input_data_all(&mut self) -> Vec<&mut [f32]> {
        (0..self.input_count())
            .map(|i| unsafe {
                let data = self.input_data(i);
                std::slice::from_raw_parts_mut(data.as_mut_ptr(), data.len())
            })
            .collect()
    }
    fn output_data(&self, index: usize) -> &[f32];
    fn output_data_all(&self) -> Vec<&[f32]> {
        (0..self.output_count())
            .map(|i| self.output_data(i))
            .collect()
    }

    fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<(), Error>;
    fn set_input_data_all(&mut self, data: &[&[f32]]) -> Result<(), Error> {
        data.iter()
            .enumerate()
            .try_for_each(|(i, data)| self.set_input_data(i, data))
    }
    fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<(), Error>;
    fn set_output_data_all(&mut self, data: &mut [&mut [f32]]) -> Result<(), Error> {
        data.iter_mut()
            .enumerate()
            .try_for_each(|(i, data)| self.set_output_data(i, data))
    }

    fn run(&mut self) -> Result<(), Error>;
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    SysError(String),

    #[error("{0}")]
    Unknown(#[from] Box<dyn std::error::Error>),
}
