if(NOT TARGET inference_engine_core)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/.. core)
endif()
