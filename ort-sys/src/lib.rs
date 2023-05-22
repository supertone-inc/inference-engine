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

impl<T, E: From<Error>> From<Result<T>> for std::result::Result<T, E> {
    fn from(result: Result<T>) -> Self {
        match result.code {
            ResultCode::Ok => Ok(result.value),
            _ => Err(result.error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Error(String);

    impl From<super::Error> for Error {
        fn from(e: super::Error) -> Self {
            unsafe {
                Self(
                    std::ffi::CStr::from_ptr(e.get_message())
                        .to_string_lossy()
                        .into(),
                )
            }
        }
    }

    type Result<T> = std::result::Result<T, Error>;

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
            match Result::from(OrtInferenceEngine::create(std::ptr::null(), 0)) {
                Ok(_) => panic!("expected error"),
                Err(Error(message)) => assert_eq!(message, "No graph was found in the protobuf."),
            }
        }
    }

    #[test]
    fn with_dynamic_shape_model() {
        unsafe {
            let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul_dynamic_shape.onnx");
            let mut engine = Result::from(OrtInferenceEngine::create(
                model_data.as_ptr() as _,
                model_data.len(),
            ))
            .unwrap();
            assert_eq!(engine.get_input_count(), 2);
            assert_eq!(engine.get_output_count(), 1);
            assert_eq!(get_input_shapes(&engine), [[0, 0], [0, 0]]);
            assert_eq!(get_output_shapes(&engine), [[0, 0]]);

            set_input_shapes(&mut engine, &[&[2, 1], &[1, 2]]);
            assert_eq!(get_input_shapes(&engine), [[2, 1], [1, 2]]);

            set_output_shapes(&mut engine, &[&[2, 2]]);
            assert_eq!(get_output_shapes(&engine), [[2, 2]]);

            let input_data = [[1., 2.], [3., 4.]];
            set_input_data(&mut engine, &[&input_data[0], &input_data[1]]);

            let mut output_data = [[0., 0., 0., 0.]];
            set_output_data(&mut engine, &mut [&mut output_data[0]]);

            Result::from(engine.run()).unwrap();
            assert_eq!(output_data, [[3., 4., 6., 8.]]);
        }
    }

    unsafe fn get_input_shapes(engine: &OrtInferenceEngine) -> Vec<Vec<usize>> {
        let mut input_shapes = vec![];
        for i in 0..engine.get_input_count() {
            let shape = engine.get_input_shape(i);
            input_shapes.push(std::slice::from_raw_parts(shape.data, shape.size).into());
        }
        input_shapes
    }

    unsafe fn set_input_shapes(engine: &mut OrtInferenceEngine, shapes: &[&[usize]]) {
        for i in 0..engine.get_input_count() {
            Result::from(engine.set_input_shape(i, shapes[i].as_ptr(), shapes[i].len())).unwrap();
        }
    }

    unsafe fn set_input_data(engine: &mut OrtInferenceEngine, data: &[&[f32]]) {
        for i in 0..engine.get_input_count() {
            Result::from(engine.set_input_data(i, data[i].as_ptr())).unwrap();
        }
    }

    unsafe fn get_output_shapes(engine: &OrtInferenceEngine) -> Vec<Vec<usize>> {
        let mut output_shapes = vec![];
        for i in 0..engine.get_output_count() {
            let shape = engine.get_output_shape(i);
            output_shapes.push(std::slice::from_raw_parts(shape.data, shape.size).into());
        }
        output_shapes
    }

    unsafe fn set_output_shapes(engine: &mut OrtInferenceEngine, shapes: &[&[usize]]) {
        for i in 0..engine.get_output_count() {
            Result::from(engine.set_output_shape(i, shapes[i].as_ptr(), shapes[i].len())).unwrap();
        }
    }

    unsafe fn set_output_data(engine: &mut OrtInferenceEngine, data: &mut [&mut [f32]]) {
        for i in 0..engine.get_output_count() {
            Result::from(engine.set_output_data(i, data[i].as_mut_ptr())).unwrap();
        }
    }
}
