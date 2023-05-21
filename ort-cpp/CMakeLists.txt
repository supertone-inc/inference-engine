cmake_minimum_required(VERSION 3.24)
project(inference-engine-ort)

option(INFERENCE_ENGINE_ORT_RUN_TESTS "Run tests" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/inference-engine-core.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/onnxruntime.cmake)

add_library(${PROJECT_NAME} STATIC ${CMAKE_CURRENT_LIST_DIR}/src/OrtInferenceEngine.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC inference-engine-core onnxruntime)

if(INFERENCE_ENGINE_ORT_RUN_TESTS)
    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/catch2.cmake)

    add_executable(${PROJECT_NAME}-test ${CMAKE_CURRENT_LIST_DIR}/src/OrtInferenceEngine.test.cpp)
    set_target_properties(${PROJECT_NAME}-test PROPERTIES CXX_STANDARD 17)
    target_link_libraries(${PROJECT_NAME}-test Catch2WithMain ${PROJECT_NAME})

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
