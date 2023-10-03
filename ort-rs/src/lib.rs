pub use inference_engine_core::*;

use inference_engine_ort_sys as sys;
use std::ffi::c_void;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct OrtInferenceEngine {
    raw: *mut c_void,
}

impl OrtInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self, Error> {
        unsafe {
            let model_data = model_data.as_ref();
            let mut raw = null_mut();

            Result::from(sys::inference_engine_ort__create_inference_engine(
                model_data.as_ptr() as _,
                model_data.len(),
                &mut raw,
            ))?;

            Ok(Self { raw })
        }
    }
}

sys::impl_inference_engine!(OrtInferenceEngine);

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        assert_matches!(
            OrtInferenceEngine::new([]),
            Err(Error::SysError(message)) if message == "No graph was found in the protobuf."
        );
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/matmul.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();

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
    fn with_dynamic_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/matmul_dynamic.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_shapes(), [[0, 0], [0, 0]]);
        assert_eq!(engine.output_shapes(), [[0, 0]]);

        engine.set_input_shapes(&[&[2, 1], &[1, 2]]).unwrap();
        assert_eq!(engine.input_shapes(), [[2, 1], [1, 2]]);

        engine.set_output_shapes(&[&[2, 2]]).unwrap();
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
