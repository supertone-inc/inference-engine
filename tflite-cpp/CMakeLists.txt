cmake_minimum_required(VERSION 3.24)
project(inference_engine_tflite)

set(DEFAULT_TENSORFLOWLITE_VERSION 2.15.0)

set(INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR CACHE PATH "")
set(INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION CACHE STRING "")
set(INFERENCE_ENGINE_TFLITE_RUN_TESTS OFF CACHE BOOL "")

if(NOT INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION)
    set(INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION ${DEFAULT_TENSORFLOWLITE_VERSION})
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
set(CMAKE_INSTALL_MESSAGE NEVER)

add_library(inference_engine_tflite STATIC src/TfLiteInferenceEngine.cpp)
set_target_properties(inference_engine_tflite PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(inference_engine_tflite PUBLIC include)

include(../core-cpp/cmake/inference_engine_core.cmake)
target_link_libraries(inference_engine_tflite PUBLIC inference_engine_core)

set(TENSORFLOWLITE_DIR ${INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_DIR})
set(TENSORFLOWLITE_VERSION ${INFERENCE_ENGINE_TFLITE_TENSORFLOWLITE_VERSION})
include(cmake/tensorflowlite.cmake)
target_link_libraries(inference_engine_tflite PUBLIC tensorflowlite)

add_custom_command(
    TARGET inference_engine_tflite
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/extern
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/lib
)
install(TARGETS inference_engine_tflite)
install(DIRECTORY ${TENSORFLOWLITE_DIR}/ DESTINATION extern)

if(INFERENCE_ENGINE_TFLITE_RUN_TESTS)
    add_executable(test_inference_engine_tflite src/TfLiteInferenceEngine.test.cpp)
    set_target_properties(test_inference_engine_tflite PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_inference_engine_tflite inference_engine_tflite)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(../core-cpp/cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_tflite Catch2WithMain)

    if(UNIX AND NOT APPLE)
        target_link_libraries(Catch2WithMain PUBLIC dl pthread)
    endif()

    add_custom_target(run_test_inference_engine_tflite
        ALL
        COMMAND test_inference_engine_tflite
        DEPENDS test_inference_engine_tflite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
