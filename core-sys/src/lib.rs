pub use inference_engine_core::*;

use std::ffi::CStr;

include!("bindings.rs");

impl From<InferenceEngineResultCode> for Result<(), Error> {
    fn from(code: InferenceEngineResultCode) -> Self {
        match code {
            InferenceEngineResultCode::Ok => Ok(()),
            InferenceEngineResultCode::Error => unsafe {
                Err(Error::SysError(
                    CStr::from_ptr(inference_engine__get_last_error_message())
                        .to_string_lossy()
                        .into(),
                ))
            },
        }
    }
}

#[macro_export]
macro_rules! impl_inference_engine {
    ($target:ty) => {
        mod r#impl {
            use super::*;
            use inference_engine_core as core;
            use inference_engine_core_sys as sys;
            use std::ptr::null;

            type Result<T> = std::result::Result<T, core::Error>;

            impl Drop for $target {
                fn drop(&mut self) {
                    unsafe {
                        Result::from(sys::inference_engine__destroy_inference_engine(self.raw))
                            .unwrap();
                    }
                }
            }

            impl InferenceEngine for $target {
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
                        sys::inference_engine__get_input_shape(
                            self.raw, index, &mut data, &mut size,
                        );
                        std::slice::from_raw_parts(data, size)
                    }
                }

                fn output_shape(&self, index: usize) -> &[usize] {
                    unsafe {
                        let mut data = null();
                        let mut size = 0;
                        sys::inference_engine__get_output_shape(
                            self.raw, index, &mut data, &mut size,
                        );
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
        }
    };
}
