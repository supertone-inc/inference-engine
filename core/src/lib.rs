pub trait InferenceEngine {
    type Error;

    fn input_count(&self) -> usize;
    fn input_shape(&self, index: usize) -> &[usize];
    fn set_input_shape(
        &mut self,
        index: usize,
        shape: impl AsRef<[usize]>,
    ) -> Result<(), Self::Error>;
    fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<(), Self::Error>;

    fn output_count(&self) -> usize;
    fn output_shape(&self, index: usize) -> &[usize];
    fn set_output_shape(
        &mut self,
        index: usize,
        shape: impl AsRef<[usize]>,
    ) -> Result<(), Self::Error>;
    fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<(), Self::Error>;

    fn run(&mut self) -> Result<(), Self::Error>;
}
