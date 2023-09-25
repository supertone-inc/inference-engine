#!/bin/bash

set -e

CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR:=.}
CMAKE_BUILD_DIR=${CMAKE_BUILD_DIR:=build}
CMAKE_CONFIG=${CMAKE_CONFIG:=Debug}
CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:=.}

INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR=$INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR
INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION=$INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION
INFERENCE_ENGINE_ORT_RUN_TESTS=${INFERENCE_ENGINE_ORT_RUN_TESTS:=ON}
INFERENCE_ENGINE_ORT_SYS_RUN_TESTS=${INFERENCE_ENGINE_ORT_SYS_RUN_TESTS:=ON}

cmake \
    -S "$CMAKE_SOURCE_DIR" \
    -B "$CMAKE_BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=$CMAKE_CONFIG \
    -D CMAKE_CONFIGURATION_TYPES=$CMAKE_CONFIG \
    -D CMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
    -D INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR="$INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR" \
    -D INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION=$INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION \
    -D INFERENCE_ENGINE_ORT_RUN_TESTS=$INFERENCE_ENGINE_ORT_RUN_TESTS \
    -D INFERENCE_ENGINE_ORT_SYS_RUN_TESTS=$INFERENCE_ENGINE_ORT_SYS_RUN_TESTS

cmake \
    --build "$CMAKE_BUILD_DIR" \
    --config $CMAKE_CONFIG \
    --parallel

cmake \
    --install "$CMAKE_BUILD_DIR" \
    --config $CMAKE_CONFIG
