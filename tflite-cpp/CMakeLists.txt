cmake_minimum_required(VERSION 3.24)
project(inference_engine_tflite)

option(INFERENCE_ENGINE_TFLITE_RUN_TESTS "Run tests" OFF)
set(INFERENCE_ENGINE_TFLITE_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/lib CACHE STRING "")

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(inference_engine_tflite STATIC src/TfliteInferenceEngine.cpp)
set_target_properties(inference_engine_tflite PROPERTIES
    CXX_STANDARD 17
    ARCHIVE_OUTPUT_DIRECTORY ${INFERENCE_ENGINE_TFLITE_LIB_DIR}
)
target_include_directories(inference_engine_tflite PUBLIC include)

include(cmake/inference-engine-core.cmake)
target_link_libraries(inference_engine_tflite PUBLIC inference-engine-core)

include(cmake/tensorflowlite.cmake)
target_link_libraries(inference_engine_tflite PUBLIC tensorflowlite)

add_custom_command(
    TARGET inference_engine_tflite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${tensorflowlite_SOURCE_DIR}/lib/*" "${INFERENCE_ENGINE_TFLITE_LIB_DIR}"
)

if(INFERENCE_ENGINE_TFLITE_RUN_TESTS)
    add_executable(test_inference_engine_tflite src/TfliteInferenceEngine.test.cpp)
    set_target_properties(test_inference_engine_tflite PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_inference_engine_tflite inference_engine_tflite)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_tflite Catch2WithMain)

    add_custom_target(run-tests
        ALL
        COMMAND test_inference_engine_tflite
        DEPENDS test_inference_engine_tflite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
