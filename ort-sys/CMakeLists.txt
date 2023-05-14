cmake_minimum_required(VERSION 3.24)
project(inference-engine-ort-sys)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../ort-cpp
    ${CMAKE_CURRENT_BINARY_DIR}/ort-cpp
)

add_library(${PROJECT_NAME} STATIC ${CMAKE_CURRENT_LIST_DIR}/src/lib.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC inference-engine-ort)
