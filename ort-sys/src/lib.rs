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
    fn create_with_invalid_model_data() {
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
    fn create() {
        unsafe {
            let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul.onnx");
            let result = OrtInferenceEngine::create(model_data.as_ptr() as _, model_data.len());
            assert_eq!(result.code, ResultCode::Ok);
        }
    }
}
