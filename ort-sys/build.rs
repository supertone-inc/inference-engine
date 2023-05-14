use const_format::formatcp;
use execute_command::status as exec;

fn main() {
    const CMAKE_SOURCE_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CMAKE_BUILD_DIR: &str = formatcp!("{CMAKE_SOURCE_DIR}/build");

    println!("cargo:rustc-env=CMAKE_SOURCE_DIR={CMAKE_SOURCE_DIR}");
    println!("cargo:rustc-env=CMAKE_BUILD_DIR={CMAKE_BUILD_DIR}");

    exec(format!(
        "cmake \
            -S {CMAKE_SOURCE_DIR} \
            -B {CMAKE_BUILD_DIR} \
            -D INFERENCE_ENGINE_ORT_BUILD_TESTS=OFF"
    ))
    .unwrap();
    exec(format!("cmake --build {CMAKE_BUILD_DIR} --parallel")).unwrap();

    println!("cargo:rustc-link-search={CMAKE_BUILD_DIR}");
    println!("cargo:rustc-link-lib=inference-engine-ort-sys");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=../ort-cpp");
    println!("cargo:rerun-if-changed=../core-cpp");
}
