cmake_minimum_required(VERSION 3.24)
project(inference_engine_ort_sys)

option(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(ort_sys STATIC src/lib.cpp)
set_target_properties(ort_sys PROPERTIES
    CXX_STANDARD 17
    ARCHIVE_OUTPUT_DIRECTORY thin
)
target_include_directories(ort_sys PUBLIC include)

add_subdirectory(../ort-cpp ort-cpp)
target_link_libraries(ort_sys PUBLIC inference_engine_ort)

include(../ort-cpp/cmake/bundle_static_library.cmake)
bundle_static_library(inference_engine_ort_sys ort_sys)

if(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS)
    add_executable(test_inference_engine_ort_sys src/lib.test.cpp)
    set_target_properties(test_inference_engine_ort_sys PROPERTIES CXX_STANDARD 17)
    target_include_directories(test_inference_engine_ort_sys PUBLIC include)
    target_link_libraries(test_inference_engine_ort_sys ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}inference_engine_ort_sys${CMAKE_STATIC_LIBRARY_SUFFIX})

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(../ort-cpp/cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_ort_sys Catch2WithMain)

    if(APPLE)
        target_link_libraries(test_inference_engine_ort_sys "-framework Foundation")
    endif()

    add_custom_target(run-tests
        ALL
        COMMAND test_inference_engine_ort_sys
        DEPENDS test_inference_engine_ort_sys
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()