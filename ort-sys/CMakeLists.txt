cmake_minimum_required(VERSION 3.24)
project(inference-engine-ort-sys)

option(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS "Run tests" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

add_library(${PROJECT_NAME} STATIC src/lib.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC include)

add_subdirectory(../ort-cpp ${CMAKE_CURRENT_BINARY_DIR}/ort-cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC inference-engine-ort)

if(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS)
    add_executable(${PROJECT_NAME}-test src/lib.test.cpp)
    set_target_properties(${PROJECT_NAME}-test PROPERTIES CXX_STANDARD 17)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(../ort-cpp/cmake/catch2.cmake)
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