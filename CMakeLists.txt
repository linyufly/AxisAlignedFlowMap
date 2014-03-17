cmake_minimum_required(VERSION 2.8)
project(AxisAlignedFlowMap)

find_package(VTK)
if (VTK_FOUND)
 	# include (${VTK_USE_FILE}) Uncommenting this line will cause an error.
 	message("VTK found")
else()
 	message(ERROR "VTK not found")
endif()

include_directories(${VTK_INCLUDE_DIRS})

find_package(CUDA)
if (CUDA_FOUND)
	include (FindCUDA)
	set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_35,code=sm_35)
	message("CUDA found")
else()
	message("CUDA not found")
endif()

cuda_add_library(AxisAlignedFlowMap SHARED
	lcsBigBlockInitializationForPositions.cu
	lcsBigBlockInitializationForVelocities.cu
	lcsBlockedTracingOfRK4.cu
	lcsCollectActiveParticlesForNewInterval.cu
	lcsCollectActiveParticlesForNewRun.cu
	lcsExclusiveScanForInt.cu
	lcsGetGroupsForBlocks.cu
	lcsGetStartOffsetInParticles.cu
	lcsInitialCellLocation.cu
	lcsRedistributeParticles.cu
	lcsTetrahedronBlockIntersection.cu

	lcs.cpp lcsGeometry.cpp lcsUnitTest.cpp lcsUtility.cpp
	lcsAxisAlignedFlowMap.cxx lcsFastAdvection.cpp lcsUnstructuredGridWithTimeVaryingPointData.cxx

	OPTIONS -arch sm_35
)

target_link_libraries(AxisAlignedFlowMap ${VTK_LIBRARIES})
