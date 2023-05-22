#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("bindings.rs");

impl Drop for Error {
    fn drop(&mut self) {
        unsafe {
            Error_Error_destructor(self as _);
        }
    }
}

impl Drop for OrtInferenceEngine {
    fn drop(&mut self) {
        unsafe {
            OrtInferenceEngine_OrtInferenceEngine_destructor(self as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpp() {
        use execute_command as exec;

        const CMAKE_SOURCE_DIR: &str = env!("CMAKE_SOURCE_DIR");
        const CMAKE_BUILD_DIR: &str = env!("CMAKE_BUILD_DIR");
        const CMAKE_CONFIG: &str = env!("CMAKE_CONFIG");

        exec::status(format!(
            "cmake \
                -S {CMAKE_SOURCE_DIR} \
                -B {CMAKE_BUILD_DIR} \
                -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
                -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
                -D INFERENCE_ENGINE_ORT_SYS_RUN_TESTS=ON"
        ))
        .unwrap();

        exec::status(format!(
            "cmake \
                --build {CMAKE_BUILD_DIR} \
                --config {CMAKE_CONFIG} \
                --parallel"
        ))
        .unwrap();
    }

    #[test]
    fn with_invalid_model_data() {
        unsafe {
            let result = OrtInferenceEngine::create(std::ptr::null(), 0);
            assert_eq!(result.code, ResultCode::Error);

            let error_message = std::ffi::CStr::from_ptr(result.error.get_message())
                .to_str()
                .unwrap();
            assert_eq!(error_message, "No graph was found in the protobuf.");
        }
    }

    #[test]
    fn with_dynamic_shape_model() {
        unsafe {
            let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul_dynamic_shape.onnx");
            let mut engine = unwrap(OrtInferenceEngine::create(
                model_data.as_ptr() as _,
                model_data.len(),
            ));
            assert_eq!(engine.get_input_count(), 2);
            assert_eq!(engine.get_output_count(), 1);

            let mut input_shapes = vec![];
            for i in 0..engine.get_input_count() {
                let shape = engine.get_input_shape(i);
                input_shapes.push(std::slice::from_raw_parts(shape.data, shape.size));
            }
            assert_eq!(input_shapes, [[0, 0], [0, 0]]);

            let mut output_shapes = vec![];
            for i in 0..engine.get_output_count() {
                let shape = engine.get_output_shape(i);
                output_shapes.push(std::slice::from_raw_parts(shape.data, shape.size));
            }
            assert_eq!(output_shapes, [[0, 0]]);

            let input_shapes = [[2, 1], [1, 2]];
            for i in 0..engine.get_input_count() {
                let shape = input_shapes[i];
                unwrap(engine.set_input_shape(i, shape.as_ptr(), shape.len()));
            }

            let mut input_shapes = vec![];
            for i in 0..engine.get_input_count() {
                let shape = engine.get_input_shape(i);
                input_shapes.push(std::slice::from_raw_parts(shape.data, shape.size));
            }
            assert_eq!(input_shapes, [[2, 1], [1, 2]]);

            let output_shapes = [[2, 2]];
            for i in 0..engine.get_output_count() {
                let shape = output_shapes[i];
                unwrap(engine.set_output_shape(i, shape.as_ptr(), shape.len()));
            }

            let mut output_shapes = vec![];
            for i in 0..engine.get_output_count() {
                let shape = engine.get_output_shape(i);
                output_shapes.push(std::slice::from_raw_parts(shape.data, shape.size));
            }
            assert_eq!(output_shapes, [[2, 2]]);

            let input_data = [[1., 2.], [3., 4.]];
            for i in 0..engine.get_input_count() {
                unwrap(engine.set_input_data(i, input_data[i].as_ptr()));
            }

            let mut output_data = [[0., 0., 0., 0.]];
            for i in 0..engine.get_output_count() {
                unwrap(engine.set_output_data(i, output_data[i].as_mut_ptr()));
            }

            unwrap(engine.run());

            assert_eq!(output_data, [[3., 4., 6., 8.]]);
        }
    }

    fn unwrap<T>(result: Result<T>) -> T {
        match result.code {
            ResultCode::Ok => result.value,
            _ => unsafe {
                let error_message = std::ffi::CStr::from_ptr(result.error.get_message())
                    .to_str()
                    .unwrap();

                panic!("{error_message}");
            },
        }
    }
}
