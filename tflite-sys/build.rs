fn main() {
    build_cpp();
    generate_bindings();
}

fn build_cpp() {
    use const_format::formatcp;
    use execute_command as exec;

    const CMAKE_SOURCE_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CMAKE_BUILD_DIR: &str = formatcp!("{CMAKE_SOURCE_DIR}/build");
    const CMAKE_CONFIG: &str = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };

    exec::status(format!(
        "cmake \
            -S '{CMAKE_SOURCE_DIR}' \
            -B '{CMAKE_BUILD_DIR}' \
            -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
            -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
            -D INFERENCE_ENGINE_TFLITE_RUN_TESTS=OFF \
            -D INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS=OFF"
    ))
    .unwrap();
    exec::status(format!(
        "cmake \
            --build '{CMAKE_BUILD_DIR}' \
            --config {CMAKE_CONFIG} \
            --parallel"
    ))
    .unwrap();

    println!("cargo:rustc-link-search={CMAKE_BUILD_DIR}");
    println!("cargo:rustc-link-search={CMAKE_BUILD_DIR}/{CMAKE_CONFIG}");
    println!("cargo:rustc-link-lib=inference_engine_tflite_sys");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }

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
        .clang_args([
            "-x",
            "c++",
            "-std=c++17",
            "-I../tflite-cpp/include",
            "-I../core-cpp/include",
        ])
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