use const_format::formatcp;
use execute_command::status as exec;

#[test]
fn cpp() {
    const CMAKE_SOURCE_DIR: &str = env!("CMAKE_SOURCE_DIR");
    const CMAKE_BUILD_DIR: &str = env!("CMAKE_BUILD_DIR");
    const ORT_CPP_BUILD_DIR: &str = formatcp!("{CMAKE_BUILD_DIR}/ort-cpp");
    const TEST_PROGRAM_NAME: &str = "inference-engine-ort-test";

    exec(format!(
        "cmake \
            -S {CMAKE_SOURCE_DIR} \
            -B {CMAKE_BUILD_DIR} \
            -D INFERENCE_ENGINE_ORT_BUILD_TESTS=ON"
    ))
    .unwrap();
    exec(format!("cmake --build {CMAKE_BUILD_DIR} --parallel")).unwrap();
    exec(format!("{ORT_CPP_BUILD_DIR}/{TEST_PROGRAM_NAME}")).unwrap();
}
