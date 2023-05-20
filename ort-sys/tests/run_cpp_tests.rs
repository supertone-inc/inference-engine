use const_format::formatcp;
use execute_command as exec;

#[test]
fn cpp() {
    const CMAKE_SOURCE_DIR: &str = env!("CMAKE_SOURCE_DIR");
    const CMAKE_BUILD_DIR: &str = env!("CMAKE_BUILD_DIR");
    const CMAKE_CONFIG: &str = env!("CMAKE_CONFIG");
    const ORT_CPP_BUILD_DIR: &str = formatcp!("{CMAKE_BUILD_DIR}/ort-cpp");
    const TEST_PROGRAM_NAME: &str = "inference-engine-ort-test";

    exec::status(format!(
        "cmake \
            -S {CMAKE_SOURCE_DIR} \
            -B {CMAKE_BUILD_DIR} \
            -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
            -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
            -D INFERENCE_ENGINE_ORT_BUILD_TESTS=ON"
    ))
    .unwrap();

    exec::status(format!(
        "cmake \
            --build {CMAKE_BUILD_DIR} \
            --config {CMAKE_CONFIG} \
            --parallel"
    ))
    .unwrap();

    if cfg!(windows) {
        #[rustfmt::skip]
            exec::status(format!("{ORT_CPP_BUILD_DIR}/{CMAKE_CONFIG}/{TEST_PROGRAM_NAME}")).unwrap();
    } else {
        #[rustfmt::skip]
            exec::status(format!("{ORT_CPP_BUILD_DIR}/{TEST_PROGRAM_NAME}")).unwrap();
    }
}
