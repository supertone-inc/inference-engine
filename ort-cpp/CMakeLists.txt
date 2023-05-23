cmake_minimum_required(VERSION 3.24)
project(inference-engine-ort)

option(INFERENCE_ENGINE_ORT_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(${PROJECT_NAME} STATIC src/OrtInferenceEngine.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC include)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/inference-engine-core.cmake)
target_link_libraries(${PROJECT_NAME} PUBLIC inference-engine-core)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/onnxruntime.cmake)
target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime)

if(INFERENCE_ENGINE_ORT_RUN_TESTS)
    add_executable(${PROJECT_NAME}-test src/OrtInferenceEngine.test.cpp)
    set_target_properties(${PROJECT_NAME}-test PROPERTIES CXX_STANDARD 17)
    target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(cmake/catch2.cmake)
    target_link_libraries(${PROJECT_NAME}-test Catch2WithMain)

    if(APPLE)
        target_link_libraries(${PROJECT_NAME}-test "-framework Foundation")
    endif()

    add_custom_target(run-tests
        ALL
        COMMAND ${PROJECT_NAME}-test
        DEPENDS ${PROJECT_NAME}-test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
