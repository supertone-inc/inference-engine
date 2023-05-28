cmake_minimum_required(VERSION 3.24)
project(inference_engine_ort)

set(DEFAULT_ONNXRUNTIME_VERSION 1.14.1)

set(INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR CACHE PATH "")
set(INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION CACHE STRING "")
set(INFERENCE_ENGINE_ORT_RUN_TESTS OFF CACHE BOOL "")

if(NOT INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION)
    set(INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION ${DEFAULT_ONNXRUNTIME_VERSION})
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
set(CMAKE_INSTALL_MESSAGE NEVER)

add_library(inference_engine_ort STATIC src/OrtInferenceEngine.cpp)
set_target_properties(inference_engine_ort PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(inference_engine_ort PUBLIC include)

include(cmake/inference_engine_core.cmake)
target_link_libraries(inference_engine_ort PUBLIC inference_engine_core)

set(ONNXRUNTIME_DIR ${INFERENCE_ENGINE_ORT_ONNXRUNTIME_DIR})
set(ONNXRUNTIME_VERSION ${INFERENCE_ENGINE_ORT_ONNXRUNTIME_VERSION})
include(cmake/onnxruntime.cmake)
target_link_libraries(inference_engine_ort PUBLIC onnxruntime)

add_custom_command(
    TARGET inference_engine_ort
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/extern
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/lib
)
install(TARGETS inference_engine_ort)
install(DIRECTORY ${ONNXRUNTIME_DIR}/ DESTINATION extern)

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

    add_custom_target(run_test_inference_engine_ort
        ALL
        COMMAND test_inference_engine_ort
        DEPENDS test_inference_engine_ort
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
