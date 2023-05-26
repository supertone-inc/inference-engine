pub use inference_engine_core::*;

use inference_engine_ort_sys as sys;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

pub struct OrtInferenceEngine(*mut c_void);

impl OrtInferenceEngine {
    pub fn new(model_data: impl AsRef<[u8]>) -> Result<Self> {
        unsafe {
            let model_data = model_data.as_ref();
            let mut model = null_mut();

            Result::from(sys::create_inference_engine(
                model_data.as_ptr() as _,
                model_data.len(),
                &mut model,
            ))?;

            Ok(OrtInferenceEngine(model))
        }
    }
}

impl InferenceEngine for OrtInferenceEngine {
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

    fn set_input_shape(&mut self, index: usize, shape: &[usize]) -> Result<()> {
        unsafe {
            Result::from(sys::set_input_shape(
                self.0,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn set_output_shape(&mut self, index: usize, shape: &[usize]) -> Result<()> {
        unsafe {
            Result::from(sys::set_output_shape(
                self.0,
                index,
                shape.as_ptr(),
                shape.len(),
            ))
        }
    }

    fn input_data(&mut self, index: usize) -> &mut [f32] {
        unsafe {
            let data = sys::get_input_data(self.0, index);
            let size = self.input_shape(index).iter().product();
            std::slice::from_raw_parts_mut(data, size)
        }
    }

    fn output_data(&self, index: usize) -> &[f32] {
        unsafe {
            let data = sys::get_output_data(self.0, index);
            let size = self.output_shape(index).iter().product();
            std::slice::from_raw_parts(data, size)
        }
    }

    fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<()> {
        unsafe { Result::from(sys::set_input_data(self.0, index, data.as_ptr())) }
    }

    fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<()> {
        unsafe { Result::from(sys::set_output_data(self.0, index, data.as_mut_ptr())) }
    }

    fn run(&mut self) -> Result<()> {
        unsafe { Result::from(sys::run(self.0)) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_invalid_model_data() {
        match OrtInferenceEngine::new([]) {
            Err(Error::SysError(message)) => {
                assert_eq!(message, "No graph was found in the protobuf.")
            }
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn with_fixed_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/matmul.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.output_count(), 1);

        assert_eq!(engine.input_shape(0), [2, 2]);
        assert_eq!(engine.input_shape(1), [2, 2]);
        assert_eq!(engine.output_shape(0), [2, 2]);

        let input_data = [[1., 2., 3., 4.], [5., 6., 7., 8.]];
        for i in 0..engine.input_count() {
            engine.set_input_data(i, &input_data[i]).unwrap();
            assert_eq!(
                engine.input_data(i).as_mut_ptr(),
                input_data[i].as_ptr() as _
            );
        }

        let mut output_data = [[0., 0., 0., 0.]];
        for i in 0..engine.output_count() {
            engine.set_output_data(i, &mut output_data[i]).unwrap();
            assert_eq!(engine.output_data(i).as_ptr(), output_data[i].as_ptr());
        }

        engine.run().unwrap();
        assert_eq!(output_data, [[19., 22., 43., 50.]]);
    }

    #[test]
    fn with_dynamic_shape_model() {
        let model_data = include_bytes!("../../ort-cpp/test-models/matmul_dynamic.onnx");
        let mut engine = OrtInferenceEngine::new(model_data).unwrap();

        assert_eq!(engine.input_count(), 2);
        assert_eq!(engine.output_count(), 1);

        assert_eq!(engine.input_shape(0), [0, 0]);
        assert_eq!(engine.input_shape(1), [0, 0]);
        assert_eq!(engine.output_shape(0), [0, 0]);

        engine.set_input_shape(0, &[2, 1]).unwrap();
        engine.set_input_shape(1, &[1, 2]).unwrap();
        assert_eq!(engine.input_shape(0), [2, 1]);
        assert_eq!(engine.input_shape(1), [1, 2]);

        engine.set_output_shape(0, &[2, 2]).unwrap();
        assert_eq!(engine.output_shape(0), [2, 2]);

        let input_data = [[1., 2.], [3., 4.]];
        for i in 0..engine.input_count() {
            engine.set_input_data(i, &input_data[i]).unwrap();
            assert_eq!(
                engine.input_data(i).as_mut_ptr(),
                input_data[i].as_ptr() as _
            );
        }

        let mut output_data = [[0., 0., 0., 0.]];
        for i in 0..engine.output_count() {
            engine.set_output_data(i, &mut output_data[i]).unwrap();
            assert_eq!(engine.output_data(i).as_ptr(), output_data[i].as_ptr());
        }

        engine.run().unwrap();
        assert_eq!(output_data, [[3., 4., 6., 8.]]);
    }
}
