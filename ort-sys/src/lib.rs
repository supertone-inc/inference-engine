#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("bindings.rs");

use inference_engine_core::Error;
use std::ffi::CStr;

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

#[cfg(test)]
mod tests {
    #[test]
    fn cpp() {
        #[cfg(not(windows))]
        execute_command::status("./test.sh").unwrap();

        #[cfg(windows)]
        execute_command::status("./test.bat").unwrap();
    }
}
