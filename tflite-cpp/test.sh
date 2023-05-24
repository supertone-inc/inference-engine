#!/bin/bash

set -e

CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR:=.}
CMAKE_BUILD_DIR=${CMAKE_BUILD_DIR:=build}
CMAKE_CONFIG=${CMAKE_CONFIG:=Debug}

cmake \
    -S $CMAKE_SOURCE_DIR \
    -B $CMAKE_BUILD_DIR \
    -D CMAKE_BUILD_TYPE=$CMAKE_CONFIG \
    -D CMAKE_CONFIGURATION_TYPES=$CMAKE_CONFIG \
    -D INFERENCE_ENGINE_TFLITE_RUN_TESTS=ON

cmake \
    --build $CMAKE_BUILD_DIR \
    --config $CMAKE_CONFIG \
    --parallel
