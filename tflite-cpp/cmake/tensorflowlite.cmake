set(TENSORFLOWLITE_VERSION 2.12.0)

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
    set(TENSORFLOWLITE_OS linux)
    set(TENSORFLOWLITE_EXTENSION .tar.gz)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(TENSORFLOWLITE_ARCH x86_64)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64)
        set(TENSORFLOWLITE_ARCH aarch64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    set(TENSORFLOWLITE_OS macos)
    set(TENSORFLOWLITE_EXTENSION .tar.gz)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(TENSORFLOWLITE_ARCH x86_64)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64)
        set(TENSORFLOWLITE_ARCH aarch64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    set(TENSORFLOWLITE_OS windows)
    set(TENSORFLOWLITE_EXTENSION .zip)

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL AMD64)
        set(TENSORFLOWLITE_ARCH x86_64)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else()
    message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
endif()

include(FetchContent)
FetchContent_Declare(
    tensorflowlite
    URL https://github.com/supertone-inc/tensorflowlite-build/releases/download/v${TENSORFLOWLITE_VERSION}/tensorflowlite-${TENSORFLOWLITE_OS}-${TENSORFLOWLITE_ARCH}-static-lib-${TENSORFLOWLITE_VERSION}${TENSORFLOWLITE_EXTENSION}
)
FetchContent_MakeAvailable(tensorflowlite)

add_library(tensorflowlite INTERFACE)
target_include_directories(tensorflowlite INTERFACE ${tensorflowlite_SOURCE_DIR}/include)
target_link_libraries(tensorflowlite INTERFACE ${tensorflowlite_SOURCE_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tensorflowlite${CMAKE_STATIC_LIBRARY_SUFFIX})
