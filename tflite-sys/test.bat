@echo off

if "%CMAKE_SOURCE_DIR%"=="" set "CMAKE_SOURCE_DIR=."
if "%CMAKE_BUILD_DIR%"=="" set "CMAKE_BUILD_DIR=build"
if "%CMAKE_CONFIG%"=="" set "CMAKE_CONFIG=Debug"
if "%CMAKE_INSTALL_PREFIX%"=="" set "CMAKE_INSTALL_PREFIX=."

cmake ^
    -S %CMAKE_SOURCE_DIR% ^
    -B %CMAKE_BUILD_DIR% ^
    -D CMAKE_BUILD_TYPE=%CMAKE_CONFIG% ^
    -D CMAKE_CONFIGURATION_TYPES=%CMAKE_CONFIG% ^
    -D INFERENCE_ENGINE_TFLITE_RUN_TESTS=ON ^
    -D INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS=ON ^
    || exit $?

cmake ^
    --build %CMAKE_BUILD_DIR% ^
    --config %CMAKE_CONFIG% ^
    --parallel ^
    || exit $?

cmake ^
    --install %CMAKE_BUILD_DIR% ^
    --prefix %CMAKE_INSTALL_PREFIX% ^
    --config %CMAKE_CONFIG% ^
    || exit $?
