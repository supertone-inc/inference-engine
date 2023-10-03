pub use inference_engine_core::*;

use inference_engine_tflite_sys as sys;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

#[derive(Debug)]
pub struct TfLiteInferenceEngine {
    raw: *mut c_void,

    #[allow(dead_code)]
    model_data: Vec<u8>,
}

impl TfLiteInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self, Error> {
        unsafe {
            let model_data = model_data.as_ref().to_owned();
            let mut raw = null_mut();

            Result::from(sys::inference_engine_tflite__create_inference_engine(
                if model_data.is_empty() {
                    null()
                } else {
                    model_data.as_ptr() as _
                },
                model_data.len(),
                &mut raw,
            ))?;

            Ok(Self { raw, model_data })
        }
    }
}

sys::impl_inference_engine!(TfLiteInferenceEngine);

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        assert_matches!(
            TfLiteInferenceEngine::new([]),
            Err(Error::SysError(message)) if message == "failed to load model"
        );
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../tflite-cpp/test-models/matmul.tflite");
        let mut engine = TfLiteInferenceEngine::new(model_data).unwrap();

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
        let mut engine = TfLiteInferenceEngine::new(model_data).unwrap();

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
