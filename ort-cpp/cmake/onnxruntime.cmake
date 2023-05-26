set(ONNXRUNTIME_VERSION 1.14.1)

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(ONNXRUNTIME_OS linux)
    set(ONNXRUNTIME_EXTENSION .tgz)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(ONNXRUNTIME_ARCH x64)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
        set(ONNXRUNTIME_ARCH aarch64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    set(ONNXRUNTIME_OS osx)
    set(ONNXRUNTIME_EXTENSION .tgz)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(ONNXRUNTIME_ARCH x86_64)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64)
        set(ONNXRUNTIME_ARCH arm64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    set(ONNXRUNTIME_OS win)
    set(ONNXRUNTIME_EXTENSION .zip)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL AMD64)
        set(ONNXRUNTIME_ARCH x64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else()
    message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
endif()

include(FetchContent)
FetchContent_Declare(
    onnxruntime
    URL https://github.com/supertone-inc/onnxruntime-build/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_OS}-${ONNXRUNTIME_ARCH}-static-lib-${ONNXRUNTIME_VERSION}${ONNXRUNTIME_EXTENSION}
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime INTERFACE)
target_include_directories(onnxruntime INTERFACE ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(onnxruntime INTERFACE ${onnxruntime_SOURCE_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnxruntime${CMAKE_STATIC_LIBRARY_SUFFIX})
