cmake_minimum_required(VERSION 2.8)
project(BuildTest)

find_package(VTK)
if (VTK_FOUND)
 	include (${VTK_USE_FILE})
 	message("VTK found")
else()
 	message(ERROR "VTK not found")
endif()

add_executable(FlowMap main.cpp)

target_link_libraries(FlowMap ${VTK_LIBRARIES} ${CMAKE_SOURCE_DIR}/libAxisAlignedFlowMap.so)
