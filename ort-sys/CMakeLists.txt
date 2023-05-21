cmake_minimum_required(VERSION 3.24)
project(inference-engine-ort-sys)

add_subdirectory(
    ../ort-cpp
    ${CMAKE_CURRENT_BINARY_DIR}/ort-cpp
)

add_library(${PROJECT_NAME} STATIC
    src/lib.cpp
    src/Error.cpp
    src/OrtInferenceEngine.cpp
    src/Result.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC inference-engine-ort)
