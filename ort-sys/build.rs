use const_format::formatcp;
use execute_command as exec;

fn main() {
    const CMAKE_SOURCE_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CMAKE_BUILD_DIR: &str = formatcp!("{CMAKE_SOURCE_DIR}/build");
    const CMAKE_CONFIG: &str = if cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };

    println!("cargo:rustc-env=CMAKE_SOURCE_DIR={CMAKE_SOURCE_DIR}");
    println!("cargo:rustc-env=CMAKE_BUILD_DIR={CMAKE_BUILD_DIR}");
    println!("cargo:rustc-env=CMAKE_CONFIG={CMAKE_CONFIG}");

    exec::status(format!(
        "cmake \
            -S {CMAKE_SOURCE_DIR} \
            -B {CMAKE_BUILD_DIR} \
            -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
            -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
            -D INFERENCE_ENGINE_ORT_BUILD_TESTS=OFF"
    ))
    .unwrap();
    exec::status(format!(
        "cmake \
            --build {CMAKE_BUILD_DIR} \
            --config {CMAKE_CONFIG} \
            --parallel"
    ))
    .unwrap();

    println!("cargo:rustc-link-search={CMAKE_BUILD_DIR}");
    println!("cargo:rustc-link-lib=inference-engine-ort-sys");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=../ort-cpp");
    println!("cargo:rerun-if-changed=../core-cpp");
}
