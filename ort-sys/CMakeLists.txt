cmake_minimum_required(VERSION 3.24)
project(inference_engine_ort_sys)

set(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS OFF CACHE BOOL "")

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
set(CMAKE_INSTALL_MESSAGE NEVER)

add_library(inference_engine_ort_sys STATIC src/lib.cpp)
set_target_properties(inference_engine_ort_sys PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(inference_engine_ort_sys PUBLIC include)

add_subdirectory(../ort-cpp ort-cpp)
target_link_libraries(inference_engine_ort_sys PUBLIC inference_engine_ort)

add_custom_command(
    TARGET inference_engine_ort_sys
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/lib
)
install(TARGETS inference_engine_ort_sys)

if(INFERENCE_ENGINE_ORT_SYS_RUN_TESTS)
    add_executable(test_inference_engine_ort_sys src/lib.test.cpp)
    set_target_properties(test_inference_engine_ort_sys PROPERTIES CXX_STANDARD 17)
    target_link_libraries(test_inference_engine_ort_sys inference_engine_ort_sys)

    set(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS ON)
    include(../ort-cpp/cmake/catch2.cmake)
    target_link_libraries(test_inference_engine_ort_sys Catch2WithMain)

    if(APPLE)
        target_link_libraries(test_inference_engine_ort_sys "-framework Foundation")
    endif()

    add_custom_target(run_test_inference_engine_ort_sys
        ALL
        COMMAND test_inference_engine_ort_sys
        DEPENDS test_inference_engine_ort_sys
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()