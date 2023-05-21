use execute_command as exec;

#[test]
fn cpp() {
    const CMAKE_SOURCE_DIR: &str = env!("CMAKE_SOURCE_DIR");
    const CMAKE_BUILD_DIR: &str = env!("CMAKE_BUILD_DIR");
    const CMAKE_CONFIG: &str = env!("CMAKE_CONFIG");

    exec::status(format!(
        "cmake \
            -S {CMAKE_SOURCE_DIR} \
            -B {CMAKE_BUILD_DIR} \
            -D CMAKE_BUILD_TYPE={CMAKE_CONFIG} \
            -D CMAKE_CONFIGURATION_TYPES={CMAKE_CONFIG} \
            -D INFERENCE_ENGINE_ORT_RUN_TESTS=ON"
    ))
    .unwrap();

    exec::status(format!(
        "cmake \
            --build {CMAKE_BUILD_DIR} \
            --config {CMAKE_CONFIG} \
            --parallel"
    ))
    .unwrap();
}
