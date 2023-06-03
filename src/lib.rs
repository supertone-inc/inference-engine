pub use inference_engine_core::*;

#[cfg(any(feature = "ort", test))]
pub use inference_engine_ort as ort;

#[cfg(any(feature = "tflite", test))]
pub use inference_engine_tflite as tflite;

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming_collision() {
        assert_matches!(
            ort::OrtInferenceEngine::new([]),
            Err(Error::SysError(message)) if message == "No graph was found in the protobuf."
        );

        assert_matches!(
            tflite::TfLiteInferenceEngine::new([]),
            Err(Error::SysError(message)) if message == "failed to load model"
        );
    }
}
