fn main() {
    build_cpp();

    #[cfg(feature = "generate-bindings")]
    generate_bindings();
}

fn build_cpp() {
    use execute_command::ExecuteCommand;
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    #[rustfmt::skip] const TENSORFLOWLITE_DIR: Option<&str> = option_env!("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR");
    #[rustfmt::skip] const TENSORFLOWLITE_VERSION: Option<&str> = option_env!("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let cmake_install_prefix = env::var("OUT_DIR").unwrap();

    #[rustfmt::skip]
    if cfg!(unix) {
        Command::new("./build.sh")
    } else {
        Command::new("./build.bat")
    }
    .env("CMAKE_BUILD_DIR", format!("{cmake_install_prefix}/build"))
    .env("CMAKE_CONFIG", if cfg!(debug_assertions) { "Debug" } else { "Release" })
    .env("CMAKE_INSTALL_PREFIX", &cmake_install_prefix)
    .env("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR", TENSORFLOWLITE_DIR.unwrap_or_default())
    .env("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION", TENSORFLOWLITE_VERSION.unwrap_or_default())
    .env("INFERENCE_ENGINE_TFLITE_RUN_TESTS", "OFF")
    .env("INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS", "OFF")
    .env("CMAKE_OPTIONS",
        match (target_os.as_str(), target_arch.as_str()) {
            ("macos", "aarch64") => {
                "-D CMAKE_OSX_ARCHITECTURES=arm64"
            }
            ("macos", "x86_64") => {
                "-D CMAKE_OSX_ARCHITECTURES=x86_64"
            }
            _ => {
                ""
            }
        }
    )
    .execute_status()
    .unwrap();

    let lib_dir = PathBuf::from(&cmake_install_prefix)
        .join("lib")
        .display()
        .to_string();
    println!("cargo:LIB_DIR={lib_dir}");
    println!("cargo:rustc-link-search={lib_dir}");
    println!("cargo:rustc-link-lib=inference_engine_core_sys");
    println!("cargo:rustc-link-lib=inference_engine_tflite_sys");
    println!("cargo:rustc-link-lib=inference_engine_tflite");

    let extern_lib_dir = PathBuf::from(&cmake_install_prefix)
        .join("extern")
        .join("lib")
        .display()
        .to_string();
    println!("cargo:EXTERN_LIB_DIR={extern_lib_dir}");
    println!("cargo:rustc-link-search={extern_lib_dir}");
    println!("cargo:rustc-link-lib=tensorflowlite");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }

    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR");
    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION");

    println!("cargo:rerun-if-changed=.");
    println!("cargo:rerun-if-changed=../core-cpp");
    println!("cargo:rerun-if-changed=../core-sys");
    println!("cargo:rerun-if-changed=../tflite-cpp");
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings() {
    use std::path::PathBuf;

    let header_path = PathBuf::from("include").join("lib.h");
    let output_path = PathBuf::from("src").join("bindings.rs");

    bindgen::Builder::default()
        .header(header_path.display().to_string())
        .allowlist_file(header_path.display().to_string())
        .blocklist_item("InferenceEngineResultCode")
        .clang_args(["-I../core-sys/include"])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .raw_line("pub use inference_engine_core_sys::*;")
        .size_t_is_usize(true)
        .generate()
        .unwrap()
        .write_to_file(output_path)
        .unwrap();

    println!("cargo:rerun-if-changed=include");
}
