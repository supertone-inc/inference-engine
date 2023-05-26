use thiserror::Error;

pub trait InferenceEngine {
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;

    fn input_shape(&self, index: usize) -> &[usize];
    fn output_shape(&self, index: usize) -> &[usize];

    fn set_input_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<(), Error>;
    fn set_output_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<(), Error>;

    fn get_input_data(&mut self, index: usize) -> &mut [f32];
    fn get_output_data(&self, index: usize) -> &[f32];

    fn set_input_data(&mut self, index: usize, data: &impl AsRef<[f32]>) -> Result<(), Error>;
    fn set_output_data(&mut self, index: usize, data: &mut impl AsMut<[f32]>) -> Result<(), Error>;

    fn run(&mut self) -> Result<(), Error>;
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    SysError(String),

    #[error("{0}")]
    Unknown(#[from] Box<dyn std::error::Error>),
}
