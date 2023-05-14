pub trait InferenceEngine {
    fn input_shapes<Size>(&self) -> &[&[Size]];

    fn output_shapes<Size>(&self) -> &[&[Size]];

    fn run<Input, Output, Error>(
        &mut self,
        inputs: impl AsRef<[Input]>,
        outputs: impl AsMut<[Output]>,
    ) -> Result<(), Error>;
}
