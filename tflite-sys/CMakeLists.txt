cmake_minimum_required(VERSION 3.24)
project(inference_engine_tflite_sys)

option(INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(inference_engine_tflite_sys STATIC src/lib.cpp)
set_target_properties(inference_engine_tflite_sys PROPERTIES CXX_STANDARD 17)
target_include_directories(inference_engine_tflite_sys PUBLIC include)

add_subdirectory(../tflite-cpp tflite-cpp)
target_link_libraries(inference_engine_tflite_sys PUBLIC inference_engine_tflite)

install(TARGETS inference_engine_tflite_sys)

if(INFERENCE_ENGINE_TFLITE_SYS_RUN_TESTS)
    add_executable(test_inference_engine_tflite_sys src/lib.test.cpp)
    set_target_properties(test_inference_engine_tflite_sys PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_inference_engine_tflite_sys inference_engine_tflite_sys)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(../tflite-cpp/cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_tflite_sys Catch2WithMain)

    if(APPLE)
        target_link_libraries(test_inference_engine_tflite_sys "-framework Foundation")
    endif()

    add_custom_target(run_test_inference_engine_tflite_sys
        ALL
        COMMAND test_inference_engine_tflite_sys
        DEPENDS test_inference_engine_tflite_sys
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()