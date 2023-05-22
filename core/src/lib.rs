pub trait InferenceEngine {
    fn input_count(&self) -> usize;
    fn input_shape(&self, index: usize) -> &[usize];
    fn set_input_shape<Error>(&self, index: usize, shape: &[usize]) -> Result<(), Error>;
    fn set_input_data<Data, Error>(
        &mut self,
        index: usize,
        data: impl AsRef<Data>,
    ) -> Result<(), Error>;

    fn output_count(&self) -> usize;
    fn output_shape(&self, index: usize) -> &[usize];
    fn set_output_shape<Error>(&self, index: usize, shape: &[usize]) -> Result<(), Error>;
    fn set_output_data<Data, Error>(
        &mut self,
        index: usize,
        data: impl AsMut<Data>,
    ) -> Result<(), Error>;

    fn run<Error>(&mut self) -> Result<(), Error>;
}
