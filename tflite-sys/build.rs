fn main() {
    build_cpp();
    generate_bindings();
}

fn build_cpp() {
    use const_format::formatcp;
    use execute_command as exec;
    use std::env;
    use std::path::PathBuf;

    #[rustfmt::skip] const TENSORFLOWLITE_DIR: Option<&str> = option_env!("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR");
    #[rustfmt::skip] const TENSORFLOWLITE_VERSION: Option<&str> = option_env!("INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION");

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
            -D INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR='{TENSORFLOWLITE_DIR}' \
            -D INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION='{TENSORFLOWLITE_VERSION}' \
            -D INFERENCE_ENGINE_TFLITE_RUN_TESTS=OFF \
            -D INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS=OFF",
        TENSORFLOWLITE_DIR = TENSORFLOWLITE_DIR.unwrap_or_default(),
        TENSORFLOWLITE_VERSION = TENSORFLOWLITE_VERSION.unwrap_or_default(),
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
    println!("cargo:rustc-link-lib=stdc++");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }

    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_TENSORFLOWLITE_DIR");
    println!("cargo:rerun-if-env-changed=INFERENCE_ENGINE_TENSORFLOWLITE_VERSION");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=src/lib.cpp");
    println!("cargo:rerun-if-changed=../tflite-cpp");
    println!("cargo:rerun-if-changed=../core-cpp");
}

fn generate_bindings() {
    use std::path::PathBuf;

    let header_path = PathBuf::from("include").join("lib.hpp");
    let output_path = PathBuf::from("src").join("bindings.rs");

    bindgen::Builder::default()
        .header(header_path.display().to_string())
        .allowlist_function("inference_engine::.*")
        .clang_args(["-x", "c++", "-std=c++17"])
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .disable_name_namespacing()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .respect_cxx_access_specs(true)
        .size_t_is_usize(true)
        .generate()
        .unwrap()
        .write_to_file(output_path)
        .unwrap();

    println!("cargo:rerun-if-changed=include");
}
