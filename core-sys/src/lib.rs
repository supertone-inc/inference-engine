include!("bindings.rs");

impl From<InferenceEngineResultCode> for Result<(), inference_engine_core::Error> {
    fn from(code: InferenceEngineResultCode) -> Self {
        match code {
            InferenceEngineResultCode::Ok => Ok(()),
            InferenceEngineResultCode::Error => unsafe {
                Err(inference_engine_core::Error::SysError(
                    std::ffi::CStr::from_ptr(inference_engine__get_last_error_message())
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
            use inference_engine_core::{Error, InferenceEngine};
            use inference_engine_core_sys as sys;
            use std::ptr::null;

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

                fn input_shapes(&self) -> Vec<&[usize]> {
                    (0..self.input_count())
                        .map(|i| self.input_shape(i))
                        .collect()
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

                fn output_shapes(&self) -> Vec<&[usize]> {
                    (0..self.output_count())
                        .map(|i| self.output_shape(i))
                        .collect()
                }

                fn set_input_shape(&mut self, index: usize, shape: &[usize]) -> Result<(), Error> {
                    unsafe {
                        Result::from(sys::inference_engine__set_input_shape(
                            self.raw,
                            index,
                            shape.as_ptr(),
                            shape.len(),
                        ))
                    }
                }

                fn set_input_shapes(&mut self, shapes: &[&[usize]]) -> Result<(), Error> {
                    shapes
                        .iter()
                        .enumerate()
                        .try_for_each(|(i, shape)| self.set_input_shape(i, shape))
                }

                fn set_output_shape(&mut self, index: usize, shape: &[usize]) -> Result<(), Error> {
                    unsafe {
                        Result::from(sys::inference_engine__set_output_shape(
                            self.raw,
                            index,
                            shape.as_ptr(),
                            shape.len(),
                        ))
                    }
                }

                fn set_output_shapes(&mut self, shapes: &[&[usize]]) -> Result<(), Error> {
                    shapes
                        .iter()
                        .enumerate()
                        .try_for_each(|(i, shape)| self.set_output_shape(i, shape))
                }

                fn input_data(&mut self, index: usize) -> &mut [f32] {
                    unsafe {
                        let data = sys::inference_engine__get_input_data(self.raw, index);
                        let size = self.input_shape(index).iter().product();
                        std::slice::from_raw_parts_mut(data, size)
                    }
                }

                fn input_data_all(&mut self) -> Vec<&mut [f32]> {
                    (0..self.input_count())
                        .map(|i| unsafe {
                            let data = self.input_data(i);
                            std::slice::from_raw_parts_mut(data.as_mut_ptr(), data.len())
                        })
                        .collect()
                }

                fn output_data(&self, index: usize) -> &[f32] {
                    unsafe {
                        let data = sys::inference_engine__get_output_data(self.raw, index);
                        let size = self.output_shape(index).iter().product();
                        std::slice::from_raw_parts(data, size)
                    }
                }

                fn output_data_all(&self) -> Vec<&[f32]> {
                    (0..self.output_count())
                        .map(|i| self.output_data(i))
                        .collect()
                }

                fn set_input_data(&mut self, index: usize, data: &[f32]) -> Result<(), Error> {
                    unsafe {
                        Result::from(sys::inference_engine__set_input_data(
                            self.raw,
                            index,
                            data.as_ptr(),
                        ))
                    }
                }

                fn set_input_data_all(&mut self, data: &[&[f32]]) -> Result<(), Error> {
                    data.iter()
                        .enumerate()
                        .try_for_each(|(i, data)| self.set_input_data(i, data))
                }

                fn set_output_data(&mut self, index: usize, data: &mut [f32]) -> Result<(), Error> {
                    unsafe {
                        Result::from(sys::inference_engine__set_output_data(
                            self.raw,
                            index,
                            data.as_mut_ptr(),
                        ))
                    }
                }

                fn set_output_data_all(&mut self, data: &mut [&mut [f32]]) -> Result<(), Error> {
                    data.iter_mut()
                        .enumerate()
                        .try_for_each(|(i, data)| self.set_output_data(i, data))
                }

                fn run(&mut self) -> Result<(), Error> {
                    unsafe { Result::from(sys::inference_engine__run(self.raw)) }
                }
            }
        }
    };
}
