pub use inference_engine_core::*;

use inference_engine_tflite_sys as sys;
use std::ffi::{c_void, CStr};
use std::ptr::{null, null_mut};
use thiserror::Error;

pub struct TfliteInferenceEngine(*mut c_void);

impl TfliteInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self> {
        unsafe {
            let model_data = model_data.as_ref();
            let mut model = null_mut();

            Result::from(sys::create_inference_engine(
                model_data.as_ptr() as _,
                model_data.len(),
                &mut model,
            ))?;

            Ok(TfliteInferenceEngine(model))
        }
    }
}

impl InferenceEngine for TfliteInferenceEngine {
    type Error = Error;

    fn input_count(&self) -> usize {
        unsafe { sys::get_input_count(self.0) }
    }

    fn output_count(&self) -> usize {
        unsafe { sys::get_output_count(self.0) }
    }

    fn input_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let mut data = null();
            let mut size = 0;
            sys::get_input_shape(self.0, index, &mut data, &mut size);
            std::slice::from_raw_parts(data, size)
        }
    }

    fn output_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let mut data = null();
            let mut size = 0;
            sys::get_output_shape(self.0, index, &mut data, &mut size);
            std::slice::from_raw_parts(data, size)
        }
    }

    fn set_input_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<()> {
        unsafe {
            let shape = shape.as_ref();
            Result::from(sys::set_input_shape(
                self.0,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn set_output_shape(&mut self, index: usize, shape: impl AsRef<[usize]>) -> Result<()> {
        unsafe {
            let shape = shape.as_ref();
            Result::from(sys::set_output_shape(
                self.0,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn set_input_data(&mut self, index: usize, data: &impl AsRef<[f32]>) -> Result<()> {
        unsafe { Result::from(sys::set_input_data(self.0, index, data.as_ref().as_ptr())) }
    }

    fn set_output_data(&mut self, index: usize, data: &mut impl AsMut<[f32]>) -> Result<()> {
        unsafe {
            Result::from(sys::set_output_data(
                self.0,
                index,
                data.as_mut().as_mut_ptr(),
            ))
        }
    }

    fn run(&mut self) -> Result<()> {
        unsafe { Result::from(sys::run(self.0)) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    SysError(String),
}

impl From<sys::ResultCode> for Error {
    fn from(code: sys::ResultCode) -> Self {
        match code {
            sys::ResultCode::Ok => panic!("Expected an error."),
            sys::ResultCode::Error => unsafe {
                Self::SysError(
                    CStr::from_ptr(sys::get_last_error_message())
                        .to_string_lossy()
                        .into(),
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        match TfliteInferenceEngine::new([]) {
            Ok(_) => panic!("Expected an error."),
            Err(Error::SysError(message)) => {
                assert_eq!(message, "failed to load model")
            }
        }
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../tflite-cpp/test-models/matmul.tflite");
        let mut engine = TfliteInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.output_count(), 1);

        assert_eq!(engine.input_shape(0), [2, 2]);
        assert_eq!(engine.input_shape(1), [2, 2]);
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
    fn with_reshaping_inputs() {
        let model_data = include_bytes!("../../tflite-cpp/test-models/matmul.tflite");
        let mut engine = TfliteInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.output_count(), 1);

        assert_eq!(engine.input_shape(0), [2, 2]);
        assert_eq!(engine.input_shape(1), [2, 2]);
        assert_eq!(engine.output_shape(0), [2, 2]);

        engine.set_input_shape(0, [2, 1]).unwrap();
        engine.set_input_shape(1, [1, 2]).unwrap();
        assert_eq!(engine.input_shape(0), [2, 1]);
        assert_eq!(engine.input_shape(1), [1, 2]);
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
