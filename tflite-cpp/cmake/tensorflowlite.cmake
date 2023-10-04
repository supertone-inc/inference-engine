set(TENSORFLOWLITE_DIR ${TENSORFLOWLITE_DIR})
set(TENSORFLOWLITE_VERSION ${TENSORFLOWLITE_VERSION})

if(NOT TENSORFLOWLITE_DIR)
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

        if(CMAKE_OSX_ARCHITECTURES)
            if(${CMAKE_OSX_ARCHITECTURES} STREQUAL x86_64)
                set(TENSORFLOWLITE_ARCH x86_64)
            elseif(${CMAKE_OSX_ARCHITECTURES} STREQUAL arm64)
                set(TENSORFLOWLITE_ARCH aarch64)
            else()
                message(FATAL_ERROR "Unsupported architecture: ${CMAKE_OSX_ARCHITECTURES}")
            endif()
        else()
            if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL x86_64)
                set(TENSORFLOWLITE_ARCH x86_64)
            elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64)
                set(TENSORFLOWLITE_ARCH aarch64)
            else()
                message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
            endif()
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
        URL https://github.com/supertone-inc/tensorflowlite-build/releases/download/v${TENSORFLOWLITE_VERSION}/tensorflowlite-${TENSORFLOWLITE_OS}-${TENSORFLOWLITE_ARCH}-static_lib-${TENSORFLOWLITE_VERSION}${TENSORFLOWLITE_EXTENSION}
    )
    FetchContent_MakeAvailable(tensorflowlite)

    set(TENSORFLOWLITE_DIR ${tensorflowlite_SOURCE_DIR})
endif()

find_library(tensorflowlite
    NAMES tensorflowlite
    PATHS ${TENSORFLOWLITE_DIR}/lib
    NO_DEFAULT_PATH
)

add_library(tensorflowlite INTERFACE)
target_include_directories(tensorflowlite INTERFACE ${TENSORFLOWLITE_DIR}/include)
target_link_directories(tensorflowlite INTERFACE ${TENSORFLOWLITE_DIR}/lib)
target_link_libraries(tensorflowlite INTERFACE ${tensorflowlite})
