# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE com_srcs ${CMAKE_CURRENT_SOURCE_DIR}/post_process/*.cpp)
set(POST_PROCESS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/post_process")
set(COM_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/common/include")

# Build each source file independently
foreach(source ${srcs})
  if(NOT (${source} MATCHES "post_process"))
    # get file name
    get_filename_component(name ${source} NAME_WE)

    # get folder name
    get_filename_component(path ${source} PATH)
    get_filename_component(folder ${path} NAME_WE)

    include_directories(${CNRT_INCLUDE_DIRS} ${POST_PROCESS_INCLUDE_DIRS}
                        ${COM_INCLUDE_DIRS})

    add_executable(${name} ${source} ${com_srcs})
    target_link_libraries(${name} ${LINKER_LIBS} ${CNRT_LIBS} OffComLib)

    # Install
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})

  endif()
endforeach()
