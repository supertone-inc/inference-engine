fn main() {
    println!(
        "cargo:LIB_DIR={}",
        std::env::var("DEP_INFERENCE_ENGINE_TFLITE_SYS_LIB_DIR").unwrap()
    );
}
