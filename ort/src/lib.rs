pub use inference_engine_core::*;

use inference_engine_ort_sys as sys;
use thiserror::Error;

pub struct OrtInferenceEngine(sys::OrtInferenceEngine);

impl OrtInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self> {
        unsafe {
            let model_data = model_data.as_ref();

            Ok(Self(Result::from(sys::OrtInferenceEngine::create(
                model_data.as_ptr() as _,
                model_data.len(),
            ))?))
        }
    }
}

impl InferenceEngine for OrtInferenceEngine {
    type Error = Error;

    fn input_count(&self) -> usize {
        unsafe { self.0.get_input_count() }
    }

    fn input_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let shape = self.0.get_input_shape(index);
            std::slice::from_raw_parts(shape.data, shape.size)
        }
    }

    fn set_input_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<()> {
        unsafe {
            let shape = shape.as_ref();
            Result::from(self.0.set_input_shape(index, shape.as_ptr(), shape.len())).and(Ok(()))
        }
    }

    fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<()> {
        unsafe { Result::from(self.0.set_input_data(index, data.as_ptr())).and(Ok(())) }
    }

    fn output_count(&self) -> usize {
        unsafe { self.0.get_output_count() }
    }

    fn output_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let shape = self.0.get_output_shape(index);
            std::slice::from_raw_parts(shape.data, shape.size)
        }
    }

    fn set_output_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<()> {
        unsafe {
            let shape = shape.as_ref();
            Result::from(self.0.set_output_shape(index, shape.as_ptr(), shape.len())).and(Ok(()))
        }
    }

    fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<()> {
        unsafe { Result::from(self.0.set_output_data(index, data.as_mut_ptr())).and(Ok(())) }
    }

    fn run(&mut self) -> Result<()> {
        unsafe { Result::from(self.0.run()).and(Ok(())) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    SysError(String),
}

impl From<sys::Error> for Error {
    fn from(e: sys::Error) -> Self {
        unsafe {
            Self::SysError(
                std::ffi::CStr::from_ptr(e.get_message())
                    .to_string_lossy()
                    .into(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        match OrtInferenceEngine::new([]) {
            Ok(_) => panic!("Expected an error."),
            Err(Error::SysError(message)) => {
                assert_eq!(message, "No graph was found in the protobuf.")
            }
        }
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();
        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.input_shape(0), [2, 2]);
        assert_eq!(engine.input_shape(1), [2, 2]);
        assert_eq!(engine.output_count(), 1);
        assert_eq!(engine.output_shape(0), [2, 2]);

        let input_data = [[1., 2., 3., 4.], [5., 6., 7., 8.]];
        engine.set_input_data(0, &input_data[0]).unwrap();
        engine.set_input_data(1, &input_data[1]).unwrap();

        let mut output_data = [[0., 0., 0., 0.]];
        engine.set_output_data(0, &mut output_data[0]).unwrap();

        engine.run().unwrap();
        assert_eq!(output_data, [[19., 22., 43., 50.]]);
    }

    #[test]
    fn with_dynamic_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul_dynamic_shape.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();
        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.output_count(), 1);
        assert_eq!(engine.input_shape(0), [0, 0]);
        assert_eq!(engine.input_shape(1), [0, 0]);
        assert_eq!(engine.output_shape(0), [0, 0]);

        engine.set_input_shape(0, [2, 1]).unwrap();
        engine.set_input_shape(1, [1, 2]).unwrap();
        assert_eq!(engine.input_shape(0), [2, 1]);
        assert_eq!(engine.input_shape(1), [1, 2]);

        engine.set_output_shape(0, [2, 2]).unwrap();
        assert_eq!(engine.output_shape(0), [2, 2]);

        let input_data = [[1., 2.], [3., 4.]];
        engine.set_input_data(0, &input_data[0]).unwrap();
        engine.set_input_data(1, &input_data[1]).unwrap();

        let mut output_data = [[0., 0., 0., 0.]];
        engine.set_output_data(0, &mut output_data[0]).unwrap();

        engine.run().unwrap();
        assert_eq!(output_data, [[3., 4., 6., 8.]]);
    }
}
