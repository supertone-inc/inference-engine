fn main() {
    build_cpp();

    #[cfg(feature = "generate-bindings")]
    generate_bindings();
}

fn build_cpp() {
    use const_format::formatcp;
    use execute_command as exec;
    use std::env;
    use std::path::PathBuf;

    #[rustfmt::skip] const ONNXRUNTIME_DIR: Option<&str> = option_env!("INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR");
    #[rustfmt::skip] const ONNXRUNTIME_VERSION: Option<&str> = option_env!("INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION");

    const CMAKE_SOURCE_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CMAKE_BUILD_DIR: &str = formatcp!("{CMAKE_SOURCE_DIR}/build");
    const CMAKE_CONFIG: &str = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };
    let cmake_install_prefix = env::var("OUT_DIR").unwrap();

    #[rustfmt::skip]
    exec::status(format!(
        "cmake \
            -S '{CMAKE_SOURCE_DIR}' \
            -B '{CMAKE_BUILD_DIR}' \
            -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
            -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
            -D CMAKE_INSTALL_PREFIX='{cmake_install_prefix}' \
            -D INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR='{ONNXRUNTIME_DIR}' \
            -D INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION={ONNXRUNTIME_VERSION} \
            -D INFERENCE_ENGINE_ORT_RUN_TESTS=OFF \
            -D INFERENCE_ENGINE_ORT_SYS_RUN_TESTS=OFF",
        ONNXRUNTIME_DIR = ONNXRUNTIME_DIR.unwrap_or_default(),
        ONNXRUNTIME_VERSION = ONNXRUNTIME_VERSION.unwrap_or_default(),
    ))
    .unwrap();
    exec::status(format!(
        "cmake \
            --build '{CMAKE_BUILD_DIR}' \
            --config {CMAKE_CONFIG} \
            --parallel"
    ))
    .unwrap();
    exec::status(format!(
        "cmake \
            --install '{CMAKE_BUILD_DIR}' \
            --config {CMAKE_CONFIG}"
    ))
    .unwrap();

    let lib_dir = PathBuf::from(&cmake_install_prefix)
        .join("lib")
        .display()
        .to_string();
    println!("cargo:LIB_DIR={lib_dir}");
    println!("cargo:rustc-link-search={lib_dir}");
    println!("cargo:rustc-link-lib=inference_engine_ort_sys");
    println!("cargo:rustc-link-lib=inference_engine_ort");

    let extern_lib_dir = PathBuf::from(&cmake_install_prefix)
        .join("extern")
        .join("lib")
        .display()
        .to_string();
    println!("cargo:EXTERN_LIB_DIR={extern_lib_dir}");
    println!("cargo:rustc-link-search={extern_lib_dir}");
    println!("cargo:rustc-link-lib=onnxruntime");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR");
    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=src/lib.cpp");
    println!("cargo:rerun-if-changed=../ort-cpp");
    println!("cargo:rerun-if-changed=../core-cpp");
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings() {
    use std::path::PathBuf;

    let header_path = PathBuf::from("include").join("lib.h");
    let output_path = PathBuf::from("src").join("bindings.rs");

    bindgen::Builder::default()
        .header(header_path.display().to_string())
        .allowlist_file(header_path.display().to_string())
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .generate()
        .unwrap()
        .write_to_file(output_path)
        .unwrap();

    println!("cargo:rerun-if-changed=include");
}
