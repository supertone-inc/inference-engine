fn main() {
    #[cfg(feature = "ort")]
    {
        println!(
            "cargo:ORT_LIB_DIR={}",
            std::env::var("DEP_INFERENCE_ENGINE_ORT_LIB_DIR").unwrap()
        );

        println!(
            "cargo:ORT_EXTERN_LIB_DIR={}",
            std::env::var("DEP_INFERENCE_ENGINE_ORT_EXTERN_LIB_DIR").unwrap()
        );
    }

    #[cfg(feature = "tflite")]
    {
        println!(
            "cargo:TFLITE_LIB_DIR={}",
            std::env::var("DEP_INFERENCE_ENGINE_TFLITE_LIB_DIR").unwrap()
        );

        println!(
            "cargo:TFLITE_EXTERN_LIB_DIR={}",
            std::env::var("DEP_INFERENCE_ENGINE_TFLITE_EXTERN_LIB_DIR").unwrap()
        );
    }
}
