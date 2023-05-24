cmake_minimum_required(VERSION 3.24)
project(inference_engine_tflite)

option(INFERENCE_ENGINE_TFLITE_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(tflite STATIC src/TfliteInferenceEngine.cpp)
set_target_properties(tflite PROPERTIES
    CXX_STANDARD 17
    ARCHIVE_OUTPUT_DIRECTORY thin
)
target_include_directories(tflite PUBLIC include)

include(cmake/inference-engine-core.cmake)
target_link_libraries(tflite PUBLIC inference-engine-core)

include(cmake/tensorflowlite.cmake)
target_link_libraries(tflite PUBLIC tensorflowlite)

include(cmake/bundle_static_library.cmake)
bundle_static_library(inference_engine_tflite tflite)

if(INFERENCE_ENGINE_TFLITE_RUN_TESTS)
    add_executable(test_inference_engine_tflite src/TfliteInferenceEngine.test.cpp)
    set_target_properties(test_inference_engine_tflite PROPERTIES CXX_STANDARD 17)
    target_include_directories(test_inference_engine_tflite PUBLIC include ../core-cpp/include)
    target_link_libraries(test_inference_engine_tflite ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}inference_engine_tflite${CMAKE_STATIC_LIBRARY_SUFFIX})

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
