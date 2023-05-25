cmake_minimum_required(VERSION 3.24)
project(inference_engine_tflite)

option(INFERENCE_ENGINE_TFLITE_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(inference_engine_tflite STATIC src/TfliteInferenceEngine.cpp)
set_target_properties(inference_engine_tflite PROPERTIES CXX_STANDARD 17)
target_include_directories(inference_engine_tflite PUBLIC include)

include(cmake/inference_engine_core.cmake)
target_link_libraries(inference_engine_tflite PUBLIC inference_engine_core)

include(cmake/tensorflowlite.cmake)
target_link_libraries(inference_engine_tflite PUBLIC tensorflowlite)

install(TARGETS inference_engine_tflite)
install(DIRECTORY ${tensorflowlite_SOURCE_DIR}/lib/ TYPE LIB)

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
