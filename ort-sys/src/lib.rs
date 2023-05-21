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
    }
}
