pub use inference_engine_core::*;

use inference_engine_tflite_sys as sys;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

pub struct TfliteInferenceEngine {
    raw: *mut c_void,

    #[allow(dead_code)]
    model_data: Vec<u8>,
}

impl TfliteInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self> {
        unsafe {
            let model_data = model_data.as_ref().to_owned();
            let mut raw = null_mut();

            Result::from(sys::inference_engine__create_inference_engine(
                model_data.as_ptr() as _,
                model_data.len(),
                &mut raw,
            ))?;

            Ok(Self { raw, model_data })
        }
    }
}

impl Drop for TfliteInferenceEngine {
    fn drop(&mut self) {
        unsafe {
            sys::inference_engine__destroy_inference_engine(self.raw);
        }
    }
}

impl InferenceEngine for TfliteInferenceEngine {
    fn input_count(&self) -> usize {
        unsafe { sys::inference_engine__get_input_count(self.raw) }
    }

    fn output_count(&self) -> usize {
        unsafe { sys::inference_engine__get_output_count(self.raw) }
    }

    fn input_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let mut data = null();
            let mut size = 0;
            sys::inference_engine__get_input_shape(self.raw, index, &mut data, &mut size);
            std::slice::from_raw_parts(data, size)
        }
    }

    fn output_shape(&self, index: usize) -> &[usize] {
        unsafe {
            let mut data = null();
            let mut size = 0;
            sys::inference_engine__get_output_shape(self.raw, index, &mut data, &mut size);
            std::slice::from_raw_parts(data, size)
        }
    }

    fn set_input_shape(&mut self, index: usize, shape: &[usize]) -> Result<()> {
        unsafe {
            Result::from(sys::inference_engine__set_input_shape(
                self.raw,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn set_output_shape(&mut self, index: usize, shape: &[usize]) -> Result<()> {
        unsafe {
            Result::from(sys::inference_engine__set_output_shape(
                self.raw,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn input_data(&mut self, index: usize) -> &mut [f32] {
        unsafe {
            let data = sys::inference_engine__get_input_data(self.raw, index);
            let size = self.input_shape(index).iter().product();
            std::slice::from_raw_parts_mut(data, size)
        }
    }

    fn output_data(&self, index: usize) -> &[f32] {
        unsafe {
            let data = sys::inference_engine__get_output_data(self.raw, index);
            let size = self.output_shape(index).iter().product();
            std::slice::from_raw_parts(data, size)
        }
    }

    fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<()> {
        unsafe {
            Result::from(sys::inference_engine__set_input_data(
                self.raw,
                index,
                data.as_ptr(),
            ))
        }
    }

    fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<()> {
        unsafe {
            Result::from(sys::inference_engine__set_output_data(
                self.raw,
                index,
                data.as_mut_ptr(),
            ))
        }
    }

    fn run(&mut self) -> Result<()> {
        unsafe { Result::from(sys::inference_engine__run(self.raw)) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        match TfliteInferenceEngine::new([0]) {
            Err(Error::SysError(message)) => {
                assert_eq!(message, "failed to load model")
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../tflite-cpp/test-models/matmul.tflite");
        let mut engine = TfliteInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_shapes(), [[2, 2], [2, 2]]);
        assert_eq!(engine.output_shapes(), [[2, 2]]);

        let input_data = [[1., 2., 3., 4.], [5., 6., 7., 8.]];
        let input_data = input_data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        engine.set_input_data_all(&input_data).unwrap();
        engine
            .input_data_all()
            .iter()
            .zip(input_data.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.as_ptr(), b.as_ptr() as _);
            });

        let mut output_data = [[0., 0., 0., 0.]];
        let mut output_data = output_data
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect::<Vec<_>>();
        engine.set_output_data_all(&mut output_data).unwrap();
        engine
            .output_data_all()
            .iter()
            .zip(output_data.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.as_ptr(), b.as_ptr() as _);
            });

        engine.run().unwrap();
        assert_eq!(output_data, [[19., 22., 43., 50.]]);
    }

    #[test]
    fn with_reshaping_inputs() {
        let model_data = include_bytes!("../../tflite-cpp/test-models/matmul.tflite");
        let mut engine = TfliteInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_shapes(), [[2, 2], [2, 2]]);
        assert_eq!(engine.output_shapes(), [[2, 2]]);

        engine.set_input_shapes(&[&[2, 1], &[1, 2]]).unwrap();
        assert_eq!(engine.input_shapes(), [[2, 1], [1, 2]]);
        assert_eq!(engine.output_shapes(), [[2, 2]]);

        let input_data = [[1., 2.], [3., 4.]];
        let input_data = input_data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        engine.set_input_data_all(&input_data).unwrap();
        engine
            .input_data_all()
            .iter()
            .zip(input_data.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.as_ptr(), b.as_ptr() as _);
            });

        let mut output_data = [[0., 0., 0., 0.]];
        let mut output_data = output_data
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect::<Vec<_>>();
        engine.set_output_data_all(&mut output_data).unwrap();
        engine
            .output_data_all()
            .iter()
            .zip(output_data.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.as_ptr(), b.as_ptr() as _);
            });

        engine.run().unwrap();
        assert_eq!(output_data, [[3., 4., 6., 8.]]);
    }
}
