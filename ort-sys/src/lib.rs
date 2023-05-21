#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("bindings.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        unsafe {
            let result = OrtInferenceEngine::create(std::ptr::null(), 0);
            assert_eq!(result.code, ResultCode::Error);

            let error_message = std::ffi::CStr::from_ptr(result.error.get_message())
                .to_str()
                .unwrap();
            assert_eq!(error_message, "No graph was found in the protobuf.");
        }

        unsafe {
            let model_data = include_bytes!("../../ort-cpp/test-models/mat_mul.onnx");
            let result = OrtInferenceEngine::create(model_data.as_ptr() as _, model_data.len());
            assert_eq!(result.code, ResultCode::Ok);
        }
    }
}
