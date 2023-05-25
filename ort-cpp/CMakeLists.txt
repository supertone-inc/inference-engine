cmake_minimum_required(VERSION 3.24)
project(inference_engine_ort)

option(INFERENCE_ENGINE_ORT_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(inference_engine_ort STATIC src/OrtInferenceEngine.cpp)
set_target_properties(inference_engine_ort PROPERTIES CXX_STANDARD 17)
target_include_directories(inference_engine_ort PUBLIC include)

include(cmake/inference-engine-core.cmake)
target_link_libraries(inference_engine_ort PUBLIC inference-engine-core)

include(cmake/onnxruntime.cmake)
target_link_libraries(inference_engine_ort PUBLIC onnxruntime)

install(TARGETS inference_engine_ort)
install(DIRECTORY ${onnxruntime_SOURCE_DIR}/lib/ TYPE LIB)

if(INFERENCE_ENGINE_ORT_RUN_TESTS)
    add_executable(test_inference_engine_ort src/OrtInferenceEngine.test.cpp)
    set_target_properties(test_inference_engine_ort PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_inference_engine_ort inference_engine_ort)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_ort Catch2WithMain)

    if(APPLE)
        target_link_libraries(test_inference_engine_ort "-framework Foundation")
    endif()

    add_custom_target(run-tests
        ALL
        COMMAND test_inference_engine_ort
        DEPENDS test_inference_engine_ort
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
