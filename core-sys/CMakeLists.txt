cmake_minimum_required(VERSION 3.24)
project(inference_engine_core_sys)

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
set(CMAKE_INSTALL_MESSAGE NEVER)

add_library(inference_engine_core_sys STATIC src/lib.cpp)
set_target_properties(inference_engine_core_sys PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(inference_engine_core_sys PUBLIC include)

include(../core-cpp/cmake/inference_engine_core.cmake)
target_link_libraries(inference_engine_core_sys PUBLIC inference_engine_core)

add_custom_command(
    TARGET inference_engine_core_sys
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_INSTALL_PREFIX}/lib
)
install(TARGETS inference_engine_core_sys)
