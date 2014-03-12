#include "lcsFastAdvection.h"
#include "lcsUnstructuredGridWithTimeVaryingPointData.h"
#include "lcsAxisAlignedFlowMap.h"
#include "lcs.h"
#include "lcsUtility.h"
#include "lcsUnitTest.h"
#include "lcsGeometry.h"

#include <vtkRectilinearGrid.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkXMLRectilinearGridWriter.h>

#include <ctime>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>
#include <limits>

//#define TET_WALK_STAT

//#define GET_PATH
//#define GET_VEL
//#define BLOCK_STAT

extern "C" void TetrahedronBlockIntersection(double *vertexPositions, int *tetrahedralConnectivities, int *queryTetrahedron, int *queryBlock, bool *queryResult,
					int numOfBlocksInY, int numOfBlocksInZ, double globalMinX, double globalMinY, double globalMinZ, double blockSize,
					double epsilon, int numOfQueries, double marginRatio);

extern "C" void InitialCellLocation(double *vertexPositions, int *tetrahedralConnectivities, int *cellLocations, int xRes, int yRes, int zRes,
					double minX, double minY, double minZ, double dx, double dy, double dz, double epsilon, int numOfCells);

extern "C" void InitializeConstantsForBlockedTracingKernelOfRK4(double *globalVertexPositions, int *globalTetrahedralConnectivities,
								int *globalTetrahedralLinks, int *startOffsetInCell, int *startOffsetInPoint, double *vertexPositionsForBig,
								double *startVelocitiesForBig, double *endVelocitiesForBig, int *blockedLocalConnectivities, int *blockedLocalLinks,
								int *blockedGlobalCellIDs, int *activeBlockList, // Map active block ID to interesting block ID
								int *blockOfGroups, int *offsetInBlocks, int *stage, double *lastPosition,
								double *k, double *nx,
								double *pastTimes, double *placesOfInterest, int *startOffsetInParticle, int *blockedActiveParticleIDList,
								int *cellLocations, int *exitCells, double hostTimeStep, double hostEpsilon
#ifdef TET_WALK_STAT
								, int *numOfTetWalks
#endif
);

extern "C" void BlockedTracingOfRK4(double startTime, double endTime, double timeStep, double epsilon, int numOfActiveBlocks, int blockSize, int sharedMemorySize, int multiple);

extern "C" void GetNumOfGroupsForBlocks(int *startOffsetInParticles, int *numOfGroupsForBlocks, int numOfActiveBlocks, int groupSize);

extern "C" void AssignGroups(int *numOfGroupsForBlocks, // It should be the prefix sum now.
				int *blockOfGroups, int *offsetInBlocks, int numOfActiveBlocks);

extern "C" void CollectActiveBlocks(int *activeParticles, int *exitCells, double *placesOfInterest, int *localTetIDs, int *blockLocations, int *interestingBlockMap,
				int *startOffsetsInLocalIDMap, int *blocksOfTets, int *localIDsOfTets, int *interestingBlockMarks, int *activeBlocks,
				int *activeBlockIndices, int *numOfActiveBlocks, // Initially 0
				int mark, int numOfActiveParticles, //int numOfStages,
				int numOfBlocksInX, int numOfBlocksInY, int numOfBlocksInZ, double globalMinX, double globalMinY, double globalMinZ,
				double blockSize, double epsilon);

extern "C" void GetNumOfParticlesByStageInBlocks(int *numOfParticlesByStageInBlocks, int *particleOrders, int *stages, int *activeParticles,
					       int *blockLocations, int *activeBlockIndices, int numOfStages, int numOfActiveParticles);

extern "C" void CollectParticlesToBlocks(int *numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
				       int *particleOrders,
				       int *stages,
				       int *activeParticles,
				       int *blockLocations,
				       int *activeBlockIndices,

				       int *blockedParticleList,
				       int numOfStages, int numOfActiveParticles);

extern "C" void CollectEveryKElement(int *input, int *output, int k, int length);

extern "C" int ExclusiveScanForInt(int *d_arr, int length);

extern "C" void InitializeScanArray(int *exitCells, int *scanArray, int length);

extern "C" void CollectActiveParticles(int *exitCells, int *scanArray, int *activeParticles, int length);

extern "C" void InitializeScanArray2(int *exitCells, int *oldActiveParticles, int *scanArray, int length);

extern "C" void CollectActiveParticles2(int *exitCells, int *oldActiveParticles, int *scanArray, int *newActiveParticles, int length);

extern "C" void BigBlockInitializationForPositions(double *globalVertexPositions, int *blockedGlobalPointIDs, int *startOffsetInPoint,
						double *vertexPositionsForBig, int numOfInterestingBlocks);

extern "C" void BigBlockInitializationForVelocities(double *globalStartVelocities, double *globalEndVelocities,	int *blockedGlobalPointIDs, int *startOffsetInPoint,
						double *startVelocitiesForBig, double *endVelocitiesForBig, int numOfInterestingBlocks);

void lcsFastAdvection::CleanAfterFlowMapComputation() {
	for (int i = 0; i < this->numOfFrames; i++)
		delete this->frames[i];
	delete [] this->frames;

	delete [] this->tetrahedralConnectivities;
	delete [] this->tetrahedralLinks;
	delete [] this->vertexPositions;

	delete [] this->queryTetrahedron;
	delete [] this->queryBlock;
	delete [] this->queryResults;

	cudaFree(this->d_tetrahedralConnectivities);
	cudaFree(this->d_vertexPositions);
	cudaFree(this->d_interestingBlockMap);
	cudaFree(this->d_startOffsetsInLocalIDMap);
	cudaFree(this->d_blocksOfTets);
	cudaFree(this->d_localIDsOfTets);

	for (int i = 0; i < this->numOfInterestingBlocks; i++)
		delete this->blocks[i];
	delete [] this->blocks;

	delete [] this->startOffsetInCell;
	delete [] this->startOffsetInPoint;

	cudaFree(this->d_vertexPositionsForBig);
	cudaFree(this->d_startVelocitiesForBig);
	cudaFree(this->d_endVelocitiesForBig);
	cudaFree(this->d_startOffsetInCell);
	cudaFree(this->d_startOffsetInPoint);
	cudaFree(this->d_localConnectivities);
	cudaFree(this->d_localLinks);
	cudaFree(this->d_globalCellIDs);
	cudaFree(this->d_globalPointIDs);
	
	delete [] this->initialCellLocations;

	cudaFree(this->d_cellLocations);

	cudaFree(this->d_activeBlockOfParticles);
	cudaFree(this->d_localTetIDs);
	cudaFree(this->d_particleOrders);
	cudaFree(this->d_blockLocations);
	cudaFree(this->d_placesOfInterest);

	for (int i = 0; i < 2; i++)
		cudaFree(this->d_activeParticles[i]);

	cudaFree(this->d_exitCells);
	cudaFree(this->d_stages);
	cudaFree(this->d_pastTimes);

	cudaFree(this->d_lastPositionForRK4);
	cudaFree(this->d_kForRK4);
	cudaFree(this->d_nxForRK4);

	for (int i = 0; i < numOfInitialActiveParticles; i++)
		delete this->particleRecords[i];
	delete [] this->particleRecords;

	//delete [] this->exitCells;

	for (int i = 0; i < 2; i++)
		cudaFree(this->d_velocities[i]);

	cudaFree(this->d_interestingBlockMarks);

	cudaFree(this->d_tetrahedralLinks);
	cudaFree(this->d_exclusiveScanArrayForInt);
	cudaFree(this->d_blockOfGroups);
	cudaFree(this->d_offsetInBlocks);
	cudaFree(this->d_numOfGroupsForBlocks);
	cudaFree(this->d_activeBlocks);
	cudaFree(this->d_activeBlockIndices);
	cudaFree(this->d_numOfActiveBlocks);
	cudaFree(this->d_startOffsetInParticles);
	cudaFree(this->d_blockedActiveParticles);
	cudaFree(this->d_numOfParticlesByStageInBlocks);

#ifdef TET_WALK_STAT
	cudaFree(this->d_numOfTetWalks);
#endif
}

void lcsFastAdvection::LoadFrames(lcsUnstructuredGridWithTimeVaryingPointData *grid) {
	this->frames = new lcs::Frame * [grid->GetNumberOfSamples()];
	this->numOfFrames = grid->GetNumberOfSamples();
	
	for (int i = 0; i < grid->GetNumberOfSamples(); i++) {
		lcs::TetrahedralGrid *temp = new lcs::TetrahedralGrid(grid, !i, i);
		this->frames[i] = new lcs::Frame;
		this->frames[i]->SetTetrahedralGrid(temp);
		this->frames[i]->SetTimePoint(grid->GetTimePoint(i));
	}
}

int lcsFastAdvection::GetBlockID(int x, int y, int z) {
	return (x * numOfBlocksInY + y) * numOfBlocksInZ + z;
}

void lcsFastAdvection::GetXYZFromBlockID(int blockID, int &x, int &y, int &z) {
	z = blockID % numOfBlocksInZ;
	blockID /= numOfBlocksInZ;
	y = blockID % numOfBlocksInY;
	x = blockID / numOfBlocksInY;
}

void lcsFastAdvection::GetXYZFromPosition(const lcs::Vector &position, int &x, int &y, int &z) {
	x = (int)((position.GetX() - globalMinX) / blockSize);
	y = (int)((position.GetY() - globalMinY) / blockSize);
	z = (int)((position.GetZ() - globalMinZ) / blockSize);
}

void lcsFastAdvection::GetTopologyAndGeometry() {
	globalNumOfCells = frames[0]->GetTetrahedralGrid()->GetNumOfCells();
	globalNumOfPoints = frames[0]->GetTetrahedralGrid()->GetNumOfVertices();

	printf("globalNumOfCells = %d, globalNumOfPoints = %d\n", globalNumOfCells, globalNumOfPoints);

	tetrahedralConnectivities = new int [globalNumOfCells * 4];
	tetrahedralLinks = new int [globalNumOfCells * 4];

	vertexPositions = new double [globalNumOfPoints * 3];
	
	frames[0]->GetTetrahedralGrid()->ReadConnectivities(tetrahedralConnectivities);
	frames[0]->GetTetrahedralGrid()->ReadLinks(tetrahedralLinks);

	frames[0]->GetTetrahedralGrid()->ReadPositions(vertexPositions);
}

void lcsFastAdvection::GetGlobalBoundingBox() {
	lcs::Vector firstPoint = frames[0]->GetTetrahedralGrid()->GetVertex(0);

	globalMaxX = globalMinX = firstPoint.GetX();
	globalMaxY = globalMinY = firstPoint.GetY();
	globalMaxZ = globalMinZ = firstPoint.GetZ();

	for (int i = 1; i < globalNumOfPoints; i++) {
		lcs::Vector point = frames[0]->GetTetrahedralGrid()->GetVertex(i);

		globalMaxX = std::max(globalMaxX, point.GetX());
		globalMinX = std::min(globalMinX, point.GetX());
		
		globalMaxY = std::max(globalMaxY, point.GetY());
		globalMinY = std::min(globalMinY, point.GetY());

		globalMaxZ = std::max(globalMaxZ, point.GetZ());
		globalMinZ = std::min(globalMinZ, point.GetZ());
	}

	printf("Global Bounding Box\n");
	printf("X: [%lf, %lf], length = %lf\n", globalMinX, globalMaxX, globalMaxX - globalMinX);
	printf("Y: [%lf, %lf], length = %lf\n", globalMinY, globalMaxY, globalMaxY - globalMinY);
	printf("Z: [%lf, %lf], length = %lf\n", globalMinZ, globalMaxZ, globalMaxZ - globalMinZ);
	printf("\n");
}

void lcsFastAdvection::CalculateNumOfBlocksInXYZ() {
	blockSize = this->GetBlockSize();

	numOfBlocksInX = (int)((globalMaxX - globalMinX) / blockSize) + 1;
	numOfBlocksInY = (int)((globalMaxY - globalMinY) / blockSize) + 1;
	numOfBlocksInZ = (int)((globalMaxZ - globalMinZ) / blockSize) + 1;
}

void lcsFastAdvection::PrepareTetrahedronBlockIntersectionQueries() {
	// Get the bounding box for every tetrahedral cell
	xLeftBound = new int [globalNumOfCells];
	xRightBound = new int [globalNumOfCells];
	yLeftBound = new int [globalNumOfCells];
	yRightBound = new int [globalNumOfCells];
	zLeftBound = new int [globalNumOfCells];
	zRightBound = new int [globalNumOfCells];

	numOfQueries = 0;
	for (int i = 0; i < globalNumOfCells; i++) {
		lcs::Tetrahedron tetrahedron = frames[0]->GetTetrahedralGrid()->GetTetrahedron(i);
		lcs::Vector firstPoint = tetrahedron.GetVertex(0);
		double localMinX, localMaxX, localMinY, localMaxY, localMinZ, localMaxZ;
		localMaxX = localMinX = firstPoint.GetX();
		localMaxY = localMinY = firstPoint.GetY();
		localMaxZ = localMinZ = firstPoint.GetZ();
		for (int j = 1; j < 4; j++) {
			lcs::Vector point = tetrahedron.GetVertex(j);
			localMaxX = std::max(localMaxX, point.GetX());
			localMinX = std::min(localMinX, point.GetX());
			localMaxY = std::max(localMaxY, point.GetY());
			localMinY = std::min(localMinY, point.GetY());
			localMaxZ = std::max(localMaxZ, point.GetZ());
			localMinZ = std::min(localMinZ, point.GetZ());
		}

		// Consider the margin
		localMaxX += this->GetMarginRatio() * blockSize;
		localMaxY += this->GetMarginRatio() * blockSize;
		localMaxZ += this->GetMarginRatio() * blockSize;

		localMinX -= this->GetMarginRatio() * blockSize;
		localMinY -= this->GetMarginRatio() * blockSize;
		localMinZ -= this->GetMarginRatio() * blockSize;

		if (localMinX < globalMinX) localMinX = globalMinX;
		if (localMinY < globalMinY) localMinY = globalMinY;
		if (localMinZ < globalMinZ) localMinZ = globalMinZ;

		if (localMaxX > globalMaxX) localMaxX = globalMaxX;
		if (localMaxY > globalMaxY) localMaxY = globalMaxY;
		if (localMaxZ > globalMaxZ) localMaxZ = globalMaxZ;

		xLeftBound[i] = (int)((localMinX - globalMinX) / blockSize);
		xRightBound[i] = (int)((localMaxX - globalMinX) / blockSize);
		yLeftBound[i] = (int)((localMinY - globalMinY) / blockSize);
		yRightBound[i] = (int)((localMaxY - globalMinY) / blockSize);
		zLeftBound[i] = (int)((localMinZ - globalMinZ) / blockSize);
		zRightBound[i] = (int)((localMaxZ - globalMinZ) / blockSize);

		numOfQueries += (xRightBound[i] - xLeftBound[i] + 1) *
				(yRightBound[i] - yLeftBound[i] + 1) *
				(zRightBound[i] - zLeftBound[i] + 1);
	}

	// Prepare host input and output arrays
	queryTetrahedron = new int [numOfQueries];
	queryBlock = new int [numOfQueries];
	queryResults = new char [numOfQueries];

	int currQuery = 0;

	for (int i = 0; i < globalNumOfCells; i++)
		for (int xItr = xLeftBound[i]; xItr <= xRightBound[i]; xItr++)
			for (int yItr = yLeftBound[i]; yItr <= yRightBound[i]; yItr++)
				for (int zItr = zLeftBound[i]; zItr <= zRightBound[i]; zItr++) {
					queryTetrahedron[currQuery] = i;
					queryBlock[currQuery] = GetBlockID(xItr, yItr, zItr);
					
					/// DEBUG ///
					if (queryBlock[currQuery] < 0 || queryBlock[currQuery] >= numOfBlocksInX * numOfBlocksInY * numOfBlocksInZ) {
						printf("incorrect block = %d\n", queryBlock[currQuery]);
						printf("%d\n", numOfBlocksInX * numOfBlocksInY * numOfBlocksInZ);
						lcs::Error("incorrect block");
					}

					currQuery++;
				}

	// Release bounding box arrays
	delete [] xLeftBound;
	delete [] xRightBound;
	delete [] yLeftBound;
	delete [] yRightBound;
	delete [] zLeftBound;
	delete [] zRightBound;
}

void lcsFastAdvection::LaunchGPUforIntersectionQueries() {
	// Create CUDA C buffer pointing to the device tetrahedralConnectivities
	err = cudaMalloc(&d_tetrahedralConnectivities, sizeof(int) * globalNumOfCells * 4);
	if (err) lcs::Error("Fail to create a buffer for device tetrahedralConnectivities");

	// Create CUDA C buffer pointing to the device vertexPositions
	err = cudaMalloc(&d_vertexPositions, sizeof(double) * globalNumOfPoints * 3);
	if (err) lcs::Error("Fail to create a buffer for device vertexPositions");

	// Create CUDA C buffer pointing to the device queryTetrahedron
	err = cudaMalloc(&d_queryTetrahedron, sizeof(int) * numOfQueries);
	if (err) lcs::Error("Fail to create a buffer for device queryTetrahedron");

	// Create CUDA C buffer pointing to the device queryBlock
	err = cudaMalloc(&d_queryBlock, sizeof(int) * numOfQueries);
	if (err) lcs::Error("Fail to create a buffer for device queryBlock");

	// Create CUDA C buffer pointing to the device queryResults (output)
	err = cudaMalloc(&d_queryResults, sizeof(bool) * numOfQueries);
	if (err) lcs::Error("Fail to create a buffer for device queryResults");

	// Copy from host to device
	err = cudaMemcpy(d_tetrahedralConnectivities, tetrahedralConnectivities, sizeof(int) * globalNumOfCells * 4, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to copy tetrahedralConnectivities");
	
	err = cudaMemcpy(d_vertexPositions, vertexPositions, sizeof(double) * globalNumOfPoints * 3, cudaMemcpyHostToDevice);	
	if (err) lcs::Error("Fail to copy vertexPositions");

	err = cudaMemcpy(d_queryTetrahedron, queryTetrahedron, sizeof(int) * numOfQueries, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to copy queryTetrahedron");

	err = cudaMemcpy(d_queryBlock, queryBlock, sizeof(int) * numOfQueries, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to copy queryBlock");

	printf("Start to use GPU to process tetrahedron-block intersection queries ...\n");
	printf("\n");

	int startTime = clock();

	TetrahedronBlockIntersection(d_vertexPositions, d_tetrahedralConnectivities, d_queryTetrahedron,
					d_queryBlock, d_queryResults, numOfBlocksInY, numOfBlocksInZ, globalMinX, globalMinY, globalMinZ,
					blockSize, this->GetEpsilonForTetBlkIntersection(), numOfQueries, this->GetMarginRatio());

	int endTime = clock();

	// Copy from device to host
	err = cudaMemcpy(queryResults, d_queryResults, sizeof(bool) * numOfQueries, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to copy queryResults from device");

	// Free d_queryResults
	cudaFree(d_queryTetrahedron);
	cudaFree(d_queryBlock);
	cudaFree(d_queryResults);

	printf("First 10 results: ");
	for (int i = 0; i < 10; i++)
		printf("%d", queryResults[i]);
	printf("\n\n");

	int sum = 0;
	for (int i = 0; i < numOfQueries; i++)
		sum += queryResults[i];
	printf("sum of queryResults[i] = %d\n", sum);

	printf("The GPU Kernel for tetrahedron-block intersection queries cost %lf sec.\n",
	       (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");

	// Unit Test for Tetrahedron-Block Intersection Kernel
	startTime = clock();

	/// DEBUG ///
	/*
	int specialTet = 6083453, specialBlk = 66;
	char specialRes = 1;
	UnitTestForTetBlkIntersection(frames[0]->GetTetrahedralGrid(),
						   blockSize, globalMinX, globalMinY, globalMinZ,
						   numOfBlocksInY, numOfBlocksInZ,
						   &specialTet, &specialBlk, &specialRes,
						   1, configure->GetEpsilonForTetBlkIntersection());
	*/
	
	if (this->GetUseUnitTestForTetBlkIntersection()) {
		UnitTestForTetBlkIntersection(frames[0]->GetTetrahedralGrid(),
						   blockSize, globalMinX, globalMinY, globalMinZ,
						   numOfBlocksInY, numOfBlocksInZ,
						   queryTetrahedron, queryBlock, queryResults,
						   numOfQueries, this->GetEpsilonForTetBlkIntersection());
		printf("\n");
	}

	endTime = clock();

	printf("The unit test cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void lcsFastAdvection::DivisionProcess() {
	// Filter out empty blocks and build interestingBlockMap
	numOfBlocks = numOfBlocksInX * numOfBlocksInY * numOfBlocksInZ;
	int *interestingBlockMap = new int [numOfBlocks];
	memset(interestingBlockMap, 255, sizeof(int) * numOfBlocks);

	err = cudaMalloc(&d_interestingBlockMap, sizeof(int) * numOfBlocks);
	if (err) lcs::Error("Fail to create device interestingBlockMap");

	numOfInterestingBlocks = 0;

	/// DEBUG ///
	/*
	for (int i = 0; i < numOfQueries; i++)
		if (queryTetrahedron[i] == 6083453 || queryTetrahedron[i] == 6083454) {
			printf("tet: %d, blk: %d, res: %d\n", queryTetrahedron[i], queryBlock[i], queryResults[i]);
		}
	*/

	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			int blockID = queryBlock[i];
			if (interestingBlockMap[blockID] != -1) continue;
			interestingBlockMap[blockID] = numOfInterestingBlocks++;
		}

	err = cudaMemcpy(d_interestingBlockMap, interestingBlockMap, sizeof(int) * numOfBlocks, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to device interestingBlockMap");

	// Count the numbers of tetrahedrons in non-empty blocks and the numbers of blocks of tetrahedrons
	int sizeOfHashMap = 0;

	int *numOfTetrahedronsInBlock, *numOfBlocksOfTetrahedron;
	int **cellsInBlock;

	numOfTetrahedronsInBlock = new int [numOfInterestingBlocks];
	memset(numOfTetrahedronsInBlock, 0, sizeof(int) * numOfInterestingBlocks);

	numOfBlocksOfTetrahedron = new int [globalNumOfCells];
	memset(numOfBlocksOfTetrahedron, 0, sizeof(int) * globalNumOfCells);

	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			numOfTetrahedronsInBlock[interestingBlockMap[queryBlock[i]]]++;
			numOfBlocksOfTetrahedron[queryTetrahedron[i]]++;
			sizeOfHashMap++;
		}

	// Initialize device arrays
	err = cudaMalloc(&d_startOffsetsInLocalIDMap, sizeof(int) * (globalNumOfCells + 1));
	if (err) lcs::Error("Fail to create device startOffsetsInLocalMap");

	err = cudaMalloc(&d_blocksOfTets, sizeof(int) * sizeOfHashMap);
	if (err) lcs::Error("Fail to create device blocksOfTets");

	err = cudaMalloc(&d_localIDsOfTets, sizeof(int) * sizeOfHashMap);
	if (err) lcs::Error("Fail to create device localIDsOfTets");

	// Initialize some work arrays
	int *startOffsetsInLocalIDMap = new int [globalNumOfCells + 1];
	
	startOffsetsInLocalIDMap[0] = 0;
	for (int i = 1; i <= globalNumOfCells; i++) {
		/// DEBUG ///
		if (numOfBlocksOfTetrahedron[i - 1] == 0) {
			printf("zero: i = %d\n", i);
			break;
		}

		startOffsetsInLocalIDMap[i] = startOffsetsInLocalIDMap[i - 1] + numOfBlocksOfTetrahedron[i - 1];
	}

	int *topOfCells = new int [globalNumOfCells];
	memset(topOfCells, 0, sizeof(int) * globalNumOfCells);

	int *blocksOfTets = new int [sizeOfHashMap];
	int *localIDsOfTets = new int [sizeOfHashMap];

	// Fill cellsInblock and build local cell ID map
	cellsInBlock = new int * [numOfInterestingBlocks];

	for (int i = 0; i < numOfInterestingBlocks; i++)
		cellsInBlock[i] = new int [numOfTetrahedronsInBlock[i]];

	int *heads = new int [numOfInterestingBlocks];
	memset(heads, 0, sizeof(int) * numOfInterestingBlocks);

	for (int i = 0; i < numOfQueries; i++)
		if (queryResults[i]) {
			int tetrahedronID = queryTetrahedron[i];
			int blockID = interestingBlockMap[queryBlock[i]];

			/// DEBUG ///
			if (blockID < 0 || blockID >= numOfInterestingBlocks) {
				printf("blockID = %d\n", blockID);
				lcs::Error("incorrect blockID");
			}

			int positionInHashMap = startOffsetsInLocalIDMap[tetrahedronID] + topOfCells[tetrahedronID];
			blocksOfTets[positionInHashMap] = queryBlock[i];
			localIDsOfTets[positionInHashMap] = heads[blockID];
			topOfCells[tetrahedronID]++;

			cellsInBlock[blockID][heads[blockID]++] = tetrahedronID;
		}

	delete [] heads;

	/// DEBUG ///
	for (int i = 0; i < globalNumOfCells; i++)
		if (startOffsetsInLocalIDMap[i] >= startOffsetsInLocalIDMap[i + 1]) {
			printf("%d %d\n", i, startOffsetsInLocalIDMap[i]);
			lcs::Error("local ID Map error");
		}

	printf("hash size = %d\n", startOffsetsInLocalIDMap[globalNumOfCells]);
	printf("sizeOfHashMap = %d\n", sizeOfHashMap);

	printf("globalNumOfCells = %d\n", globalNumOfCells);

	// Fill some device arrays
	err = cudaMemcpy(d_startOffsetsInLocalIDMap, startOffsetsInLocalIDMap, sizeof(int) * (globalNumOfCells + 1), cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to device startOffsetsInLocalIDMap");

	err = cudaMemcpy(d_blocksOfTets, blocksOfTets, sizeof(int) * sizeOfHashMap, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to device blocksOfTets");

	err = cudaMemcpy(d_localIDsOfTets, localIDsOfTets, sizeof(int) * sizeOfHashMap, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to device localIDsOfTets");

	// Delete some work arrays
	delete [] startOffsetsInLocalIDMap;
	delete [] topOfCells;
	delete [] blocksOfTets;
	delete [] localIDsOfTets;
	delete [] interestingBlockMap;

	// Initialize blocks and release cellsInBlock and numOfTetrahedronsInBlock
	blocks = new lcs::BlockRecord * [numOfInterestingBlocks];
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		blocks[i] = new lcs::BlockRecord();
		blocks[i]->SetLocalNumOfCells(numOfTetrahedronsInBlock[i]);
		blocks[i]->CreateGlobalCellIDs(cellsInBlock[i]);
		delete [] cellsInBlock[i];
	}
	delete [] cellsInBlock;
	delete [] numOfTetrahedronsInBlock;

	// Initialize work arrays
	int *cellMarks = new int [globalNumOfCells];
	int *pointMarks = new int [globalNumOfPoints];
	int *localPointIDs = new int [globalNumOfPoints];
	int *localCellIDs = new int [globalNumOfCells];
	int *pointList = new int [globalNumOfPoints];
	int *tempConnectivities = new int [globalNumOfCells * 4];
	int *tempLinks = new int [globalNumOfCells * 4];
	int markCount = 0;

	memset(cellMarks, 0, sizeof(int) * globalNumOfCells);
	memset(pointMarks, 0, sizeof(int) * globalNumOfPoints);
	
	// Process blocks
	int smallEnoughBlocks = 0;
	maxSharedMemoryRequired = 0;

	//canFitInSharedMemory = new bool [numOfInterestingBlocks];

	for (int i = 0; i < numOfInterestingBlocks; i++) {
		markCount++;
		int population = 0;

		// Get local points
		for (int j = 0; j < blocks[i]->GetLocalNumOfCells(); j++) {
			int globalCellID = blocks[i]->GetGlobalCellID(j);
			cellMarks[globalCellID] = markCount;
			localCellIDs[globalCellID] = j;

			for (int k = 0; k < 4; k++) {
				int globalPointID = tetrahedralConnectivities[(globalCellID << 2) + k];
				if (globalPointID == -1 || pointMarks[globalPointID] == markCount) continue;
				pointMarks[globalPointID] = markCount;
				localPointIDs[globalPointID] = population;
				pointList[population++] = globalPointID;
			}
		}

		blocks[i]->SetLocalNumOfPoints(population);
		blocks[i]->CreateGlobalPointIDs(pointList);

		// Mark whether the block can fit into the shared memory
		int currentBlockMemoryCost = blocks[i]->EvaluateNumOfBytes();

		//if (currentBlockMemoryCost <= configure->GetSharedMemoryKilobytes() * 1024) smallEnoughBlocks++;
		if (currentBlockMemoryCost <= this->GetMaxSharedMemoryPerSM()) smallEnoughBlocks++;
		if (currentBlockMemoryCost <= this->GetMaxSharedMemoryPerSM() && currentBlockMemoryCost > maxSharedMemoryRequired) maxSharedMemoryRequired = currentBlockMemoryCost;

		//if (currentBlockMemoryCost <= configure->GetSharedMemoryKilobytes() * 1024) {
		//	smallEnoughBlocks++;
		//	canFitInSharedMemory[i] = true;
		//} else
		//	canFitInSharedMemory[i] = false;

		// Calculate the local connectivity and link
		for (int j = 0; j < blocks[i]->GetLocalNumOfCells(); j++) {
			int globalCellID = blocks[i]->GetGlobalCellID(j);

			// Fill tempConnectivities
			for (int k = 0; k < 4; k++) {
				int globalPointID = tetrahedralConnectivities[(globalCellID << 2) + k];
				int localPointID;
				if (globalPointID != -1 && pointMarks[globalPointID] == markCount)
					localPointID = localPointIDs[globalPointID];
				else localPointID = -1;
				tempConnectivities[(j << 2) + k] = localPointID;
			}

			// Fill tempLinks
			for (int k = 0; k < 4; k++) {
				int globalNeighborID = tetrahedralLinks[(globalCellID << 2) + k];
				int localNeighborID;
				if (globalNeighborID != -1 && cellMarks[globalNeighborID] == markCount)
					localNeighborID = localCellIDs[globalNeighborID];
				else localNeighborID = -1;
				tempLinks[(j << 2) + k] = localNeighborID;
			}
		}

		blocks[i]->CreateLocalConnectivities(tempConnectivities);
		blocks[i]->CreateLocalLinks(tempLinks);
	}
	
	printf("Division is done. smallEnoughBlocks = %d\n", smallEnoughBlocks);
	printf("maxSharedMemoryRequired = %d\n", maxSharedMemoryRequired);
	printf("\n");

	// Select big blocks
	//int *bigBlocks = new int [numOfInterestingBlocks];
	//numOfBigBlocks = 0;
	//for (int i = 0; i < numOfInterestingBlocks; i++)
	//	if (!canFitInSharedMemory[i])
	//		bigBlocks[numOfBigBlocks++] = i;

	//err = cudaMalloc(&d_bigBlocks, sizeof(int) * numOfBigBlocks);
	//if (err) lcs::Error("Fail to create device bigBlocks");

	//err = cudaMemcpy(d_bigBlocks, bigBlocks, sizeof(int) * numOfBigBlocks, cudaMemcpyHostToDevice);
	//if (err) lcs::Error("Fail to write to d_bigBlockFail to write to d_bigBlocks");

	//delete [] bigBlocks;

	// Release work arrays
	delete [] cellMarks;
	delete [] pointMarks;
	delete [] localPointIDs;
	delete [] localCellIDs;
	delete [] pointList;
	delete [] tempConnectivities;
	delete [] tempLinks;

	// Some statistics
	int minPos = globalNumOfCells, maxPos = 0;
	int numOfUnder100 = 0, numOfUnder200 = 0;

	for (int i = 0; i < numOfInterestingBlocks; i++) {
		maxPos = std::max(maxPos, blocks[i]->GetLocalNumOfCells());
		minPos = std::min(minPos, blocks[i]->GetLocalNumOfCells());
		numOfUnder100 += blocks[i]->GetLocalNumOfCells() < 100;
		numOfUnder200 += blocks[i]->GetLocalNumOfCells() < 200;
	}
	
	printf("Statistics\n");
	printf("The number of blocks is %d.\n", numOfBlocks);
	printf("The number of non-zero blocks is %d.\n", numOfInterestingBlocks);
	printf("The number of under-100 blocks is %d.\n", numOfUnder100);
	printf("The number of under-200 blocks is %d.\n", numOfUnder200);
	printf("The maximum number of tetrahedrons in a block is %d.\n", maxPos);
	printf("The minimum non-zero number of tetrahedrons in a block is %d.\n", minPos);
	printf("\n");
}

void lcsFastAdvection::StoreBlocksInDevice() {
	// Initialize start offsets in cells and points
	startOffsetInCell = new int [numOfInterestingBlocks + 1];
	startOffsetInPoint = new int [numOfInterestingBlocks + 1];
	startOffsetInCell[0] = 0;
	startOffsetInPoint[0] = 0;

	// Calculate start offsets
	int maxNumOfCells = 0, maxNumOfPoints = 0;
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		startOffsetInCell[i + 1] = startOffsetInCell[i] + blocks[i]->GetLocalNumOfCells();
		startOffsetInPoint[i + 1] = startOffsetInPoint[i] + blocks[i]->GetLocalNumOfPoints();

		maxNumOfCells += blocks[i]->GetLocalNumOfCells();
		maxNumOfPoints += blocks[i]->GetLocalNumOfPoints();
	}

	printf("Total number of cells in all the blocks is %d.\n", startOffsetInCell[numOfInterestingBlocks]);
	printf("Total number of points in all the blocks is %d.\n", startOffsetInPoint[numOfInterestingBlocks]);
	printf("\n");

	//Create d_canFitInSharedMemory
	//err = cudaMalloc(&d_canFitInSharedMemory, sizeof(bool) * numOfInterestingBlocks);
	//if (err) lcs::Error("Fail to create a buffer for device canFitInSharedMemory");

	// Create d_vertexPositionsForBig
	err = cudaMalloc(&d_vertexPositionsForBig, sizeof(double) * 3 * maxNumOfPoints);
	if (err) lcs::Error("Fail to create a buffer for device vertexPositionsForBig");
	
	// Create d_startVelocitiesForBig
	err = cudaMalloc(&d_startVelocitiesForBig, sizeof(double) * 3 * maxNumOfPoints);
	if (err) lcs::Error("Fail to create a buffer for device startVelocitiesForBig");

	// Create d_endVelocitiesForBig
	err = cudaMalloc(&d_endVelocitiesForBig, sizeof(double) * 3 * maxNumOfPoints);
	if (err) lcs::Error("Fail to create a buffer for device endVelocitiesForBig");

	// Create d_startOffsetInCell
	err = cudaMalloc(&d_startOffsetInCell, sizeof(int) * (numOfInterestingBlocks + 1));
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInCell");

	// Create d_startOffsetInCellForBig
	//err = cudaMalloc(&d_startOffsetInCellForBig, sizeof(int) * (numOfInterestingBlocks + 1));
	//if (err) lcs::Error("Fail to create a buffer for device startOffsetInCellForBig");

	// Create d_startOffsetInPoint
	err = cudaMalloc(&d_startOffsetInPoint, sizeof(int) * (numOfInterestingBlocks + 1));
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInPoint");

	// Create d_startOffsetInPointForBig
	//err = cudaMalloc(&d_startOffsetInPointForBig, sizeof(int) * (numOfInterestingBlocks + 1));
	//if (err) lcs::Error("Fail to create a buffer for device startOffsetInPointForBig");

	// Create d_localConnectivities
	err = cudaMalloc(&d_localConnectivities, sizeof(int) * startOffsetInCell[numOfInterestingBlocks] * 4);
	if (err) lcs::Error("Fail to create a buffer for device localConnectivities");

	// Create d_localLinks
	err = cudaMalloc(&d_localLinks, sizeof(int) * startOffsetInCell[numOfInterestingBlocks] * 4);
	if (err) lcs::Error("Fail to create a buffer for device localLinks");

	// Create d_globalCellIDs
	err = cudaMalloc(&d_globalCellIDs, sizeof(int) * startOffsetInCell[numOfInterestingBlocks]);
	if (err) lcs::Error("Fail to create a buffer for device globalCellIDs");

	// Create d_globalPointIDs
	err = cudaMalloc(&d_globalPointIDs, sizeof(int) * startOffsetInPoint[numOfInterestingBlocks]);
	if (err) lcs::Error("Fail to create a buffer for device globalPointIDs");

	// Fill d_canFitInSharedMemory
	//err = cudaMemcpy(d_canFitInSharedMemory, canFitInSharedMemory, sizeof(bool) * numOfInterestingBlocks, cudaMemcpyHostToDevice);
	//if (err) lcs::Error("Fail to write-to-device for d_canFitInSharedMemory");

	// Fill d_startOffsetInCell
	err = cudaMemcpy(d_startOffsetInCell, startOffsetInCell, sizeof(int) * (numOfInterestingBlocks + 1), cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write-to-device for d_startOffsetInCell");

	// Fill d_startOffsetInCellForBig
	//err = cudaMemcpy(d_startOffsetInCellForBig, startOffsetInCellForBig, sizeof(int) * (numOfInterestingBlocks + 1), cudaMemcpyHostToDevice);
	//if (err) lcs::Error("Fail to write-to-device for d_startOffsetInCellForBig");

	// Fill d_startOffsetInPoint
	err = cudaMemcpy(d_startOffsetInPoint, startOffsetInPoint, sizeof(int) * (numOfInterestingBlocks + 1), cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write-to-device for d_startOffsetInPoint");

	// Fill d_startOffsetInPointForBig
	//err = cudaMemcpy(d_startOffsetInPointForBig, startOffsetInPointForBig, sizeof(int) * (numOfInterestingBlocks + 1), cudaMemcpyHostToDevice);
	//if (err) lcs::Error("Fail to write-to-device for d_startOffsetInPointForBig");

	// Fill d_localConnectivities
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currLocalConnectivities = blocks[i]->GetLocalConnectivities();

		// Enqueue write-to-device
		err = cudaMemcpy(d_localConnectivities + startOffsetInCell[i] * 4, currLocalConnectivities, length * 4 * sizeof(int), cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_localConnectivities");
	}

	// Fill d_localLinks
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currLocalLinks = blocks[i]->GetLocalLinks();

		// Enqueue write-to-device
		err = cudaMemcpy(d_localLinks + startOffsetInCell[i] * 4, currLocalLinks, length * 4 * sizeof(int), cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_localLinks");
	}

	// Fill d_globalCellIDs
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInCell[i + 1] - startOffsetInCell[i];

		if (!length) continue;

		int *currGlobalCellIDs = blocks[i]->GetGlobalCellIDs();

		// Enqueue write-to-device
		err = cudaMemcpy(d_globalCellIDs + startOffsetInCell[i], currGlobalCellIDs, length * sizeof(int), cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_globalCellIDs");
	}

	// Fill d_globalPointIDs
	for (int i = 0; i < numOfInterestingBlocks; i++) {
		int length = startOffsetInPoint[i + 1] - startOffsetInPoint[i];

		if (!length) continue;

		int *currGlobalPointIDs = blocks[i]->GetGlobalPointIDs();

		// Enqueue write-to-device
		err = cudaMemcpy(d_globalPointIDs + startOffsetInPoint[i], currGlobalPointIDs, length * sizeof(int), cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_globalPointIDs");
	}
}

void lcsFastAdvection::Division() {
	// Prepare queries
	PrepareTetrahedronBlockIntersectionQueries();

	// Launch GPU to solve queries
	LaunchGPUforIntersectionQueries();
	
	// Main process of division
	DivisionProcess();

	// Store blocks in the global memory of device
	StoreBlocksInDevice();
}

void lcsFastAdvection::AAInitialCellLocation() {
	printf("Start to use GPU to process initial cell location ...\n");
	printf("\n");

	double range[2];

	this->caller->GetXRange(range);
	double minX = range[0];
	double maxX = range[1];

	this->caller->GetYRange(range);
	double minY = range[0];
	double maxY = range[1];

	this->caller->GetZRange(range);
	double minZ = range[0];
	double maxZ = range[1];

	int dimensions[3];

	this->caller->GetDimensions(dimensions);
	int xRes = dimensions[0];
	int yRes = dimensions[1];
	int zRes = dimensions[2];

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);
	initialCellLocations = new int [numOfGridPoints];

	// Create OpenCL buffer pointing to the device cellLocations (output)
	err = cudaMalloc(&d_cellLocations, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to create a buffer for device cellLocations");

	// Initialize d_cellLocations to -1 arrays
	err = cudaMemset(d_cellLocations, 255, sizeof(int) * numOfGridPoints);
	if (err) lcs::Error("Fail to initialize d_cellLocations");

	int startTime = clock();

	InitialCellLocation(d_vertexPositions, d_tetrahedralConnectivities, d_cellLocations, xRes, yRes, zRes,
			minX, minY, minZ, dx, dy, dz, this->GetEpsilon(), globalNumOfCells);

	int endTime = clock();

	// Copy from device to host
	err = cudaMemcpy(initialCellLocations, d_cellLocations, sizeof(int) * numOfGridPoints, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to get initialCellLocations");

	// Delete d_cellLocations
	cudaFree(d_cellLocations);

	/// DEBUG ///
	//FILE *locationFile = fopen("lcsInitialLocations.txt", "w");
	//for (int i = 0; i < numOfGridPoints; i++)
	//	if (initialCellLocations[i] != -1)
	//		fprintf(locationFile, "%d %d\n", i, initialCellLocations[i]);
	//fclose(locationFile);
	printf("First 10 results: ");
	for (int i = 0; i < 10; i++) {
		if (i) printf(" ");
		printf("%d", initialCellLocations[i]);
	}
	printf("\n\n");

	printf("The GPU Kernel for initial cell locations cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");

	// Unit Test for Initial Cell Location Kernel
	startTime = clock();

	if (this->GetUseUnitTestForInitialCellLocation()) {
		lcs::UnitTestForInitialCellLocations(frames[0]->GetTetrahedralGrid(),
						     xRes, yRes, zRes,
						     minX, minY, minZ,
						     dx, dy, dz,
						     initialCellLocations,
						     this->GetEpsilon());
		printf("\n");
	}

	endTime = clock();

	printf("The unit test cost %lf sec.\n", (endTime - startTime) * 1.0 / CLOCKS_PER_SEC);
	printf("\n");
}

void lcsFastAdvection::InitializeParticleRecordsInDevice() {
	/// DEBUG ///
	printf("In the beginning of InitializeParticleRecordsInDevice(): numOfInitialActiveParticles = %d\n", numOfInitialActiveParticles);

	// Initialize activeBlockOfParticles
	err = cudaMalloc(&d_activeBlockOfParticles, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device activeBlockOfParticles");

	// Initialize localTetIDs
	err = cudaMalloc(&d_localTetIDs, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device localTetIDs");

	// Initialize particleOrders
	err = cudaMalloc(&d_particleOrders, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device particleOrders");

	// Initialize blockLocations
	err = cudaMalloc(&d_blockLocations, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device blockLocations");

	// Initialize d_placesOfInterest (Another part is in lastPositions initialization)
	err = cudaMalloc(&d_placesOfInterest, sizeof(double) * 3 * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device placesOfInterest");	

	// Initialize d_activeParticles[2]
	for (int i = 0; i < 2; i++) {
		err = cudaMalloc(&d_activeParticles[i], sizeof(int) * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device activeParticles");
	}

	// Initialize d_exitCells
	err = cudaMalloc(&d_exitCells, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device exitCells");

	err = cudaMemcpy(d_exitCells, exitCells, sizeof(int) * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write-to-device for d_exitCells");

	// Initialize d_stage
	err = cudaMalloc(&d_stages, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device stages");

	err = cudaMemset(d_stages,  0, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to clear d_stage");

	// Initialize d_pastTimes
	err = cudaMalloc(&d_pastTimes, sizeof(double) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device pastTimes");

	err = cudaMemset(d_pastTimes, 0, sizeof(double) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to clear d_pastTimes");

	// Initialize some integration-specific device arrays
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {	
		// Initialize d_lastPositionForRK4
		double *lastPosition = new double [numOfInitialActiveParticles * 3];
		for (int i = 0; i < numOfInitialActiveParticles; i++) {
			lcs::ParticleRecordDataForRK4 *data = (lcs::ParticleRecordDataForRK4 *)particleRecords[i]->GetData();
			lcs::Vector point = data->GetLastPosition();
			double x = point.GetX();
			double y = point.GetY();
			double z = point.GetZ();
			lastPosition[i * 3] = x;
			lastPosition[i * 3 + 1] = y;
			lastPosition[i * 3 + 2] = z;
		}

		
		err = cudaMalloc(&d_lastPositionForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device lastPosition for RK4");

		err = cudaMemcpy(d_lastPositionForRK4, lastPosition, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_lastPositionForRK4");

		// Additional work of placesOfInterest initialization
		err = cudaMemcpy(d_placesOfInterest, lastPosition, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_placesOfInterest");

		// Initialize d_kForRK4
		err = cudaMalloc(&d_kForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device k for RK4");

		// Initialize d_nxForRK4
		err = cudaMalloc(&d_nxForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		if (err) lcs::Error("Fail to create a buffer for device nx for RK4");

		err = cudaMemcpy(d_nxForRK4, lastPosition, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyHostToDevice);
		if (err) lcs::Error("Fail to write-to-device for d_nxForRK4");

		// Initialize d_k2ForRK4
		//err = cudaMalloc(&d_k2ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		//if (err) lcs::Error("Fail to create a buffer for device k2 for RK4");

		// Initialize d_k3ForRK4
		//err = cudaMalloc(&d_k3ForRK4, sizeof(double) * 3 * numOfInitialActiveParticles);
		//if (err) lcs::Error("Fail to create a buffer for device k3 for RK4");
	} break;
	}

	// Release some arrays
	delete [] exitCells;
}

void lcsFastAdvection::FABigBlockInitializationForPositions() {
	BigBlockInitializationForPositions(d_vertexPositions, d_globalPointIDs, d_startOffsetInPoint, d_vertexPositionsForBig, numOfInterestingBlocks);
}

void lcsFastAdvection::FABigBlockInitializationForVelocities(int currStartVIndex) {
	BigBlockInitializationForVelocities(d_velocities[currStartVIndex], d_velocities[1 - currStartVIndex], d_globalPointIDs, d_startOffsetInPoint,
			     		d_startVelocitiesForBig, d_endVelocitiesForBig, numOfInterestingBlocks);
}

void lcsFastAdvection::LaunchBlockedTracingKernel(int numOfWorkGroups, double beginTime, double finishTime, int blockSize, int sharedMemorySize, int multiple) {
	//printf("Start to use GPU to process blocked tracing ...\n");
	//printf("\n");

	double startTime = lcs::GetCurrentTimeInSeconds();

	BlockedTracingOfRK4(/*d_vertexPositions, d_tetrahedralConnectivities,
				d_tetrahedralLinks, d_startOffsetInCell, d_startOffsetInPoint, d_vertexPositionsForBig, d_startVelocitiesForBig, d_endVelocitiesForBig, 
				d_localConnectivities, d_localLinks, d_globalCellIDs, d_activeBlocks, // Map active block ID to interesting block ID
				d_blockOfGroups, d_offsetInBlocks, d_stages, d_lastPositionForRK4, d_k1ForRK4, d_k2ForRK4, d_k3ForRK4, d_pastTimes, d_placesOfInterest,
				d_startOffsetInParticles, d_blockedActiveParticles, d_localTetIDs, d_exitCells,*/
				beginTime, finishTime, this->caller->GetTimeStep(), this->GetEpsilon(), numOfWorkGroups, blockSize, sharedMemorySize, multiple);

	double endTime = lcs::GetCurrentTimeInSeconds();

	/// DEBUG ///
	kernelSum += endTime - startTime;
	kernelSumInInterval += endTime - startTime;

	//printf("The GPU Kernel for blocked tracing cost %lf sec.\n", endTime - startTime);
	//printf("\n");
}

void lcsFastAdvection::InitializeInitialActiveParticles() {
	// Initialize particleRecord
	double range[2];

	this->caller->GetXRange(range);
	double minX = range[0];
	double maxX = range[1];

	this->caller->GetYRange(range);
	double minY = range[0];
	double maxY = range[1];

	this->caller->GetZRange(range);
	double minZ = range[0];
	double maxZ = range[1];

	int dimensions[3];

	this->caller->GetDimensions(dimensions);
	int xRes = dimensions[0];
	int yRes = dimensions[1];
	int zRes = dimensions[2];

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	int numOfGridPoints = (xRes + 1) * (yRes + 1) * (zRes + 1);

	// Get numOfInitialActiveParticles
	numOfInitialActiveParticles = 0;
	for (int i = 0; i < numOfGridPoints; i++)
		if (initialCellLocations[i] != -1) numOfInitialActiveParticles++;

	if (!numOfInitialActiveParticles)
		lcs::Error("There is no initial active particle for tracing.");

	// Initialize particleRecords
	particleRecords = new lcs::ParticleRecord * [numOfInitialActiveParticles];

	printf("after new particle records\n");

	int idx = -1, activeIdx = -1;
	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				idx++;

				if (initialCellLocations[idx] == -1) continue;

				activeIdx++;

				switch (lcs::ParticleRecord::GetDataType()) {
				case lcs::ParticleRecord::RK4: {
					lcs::ParticleRecordDataForRK4 *data = new lcs::ParticleRecordDataForRK4();
					data->SetLastPosition(lcs::Vector(minX + i * dx, minY + j * dy, minZ + k * dz));
					particleRecords[activeIdx] = new
								     lcs::ParticleRecord(lcs::ParticleRecordDataForRK4::COMPUTING_K1,
								     idx, data);
				} break;
				}
			}

	printf("after 3 nested loops %d %d\n", activeIdx, numOfInitialActiveParticles);

	// Initialize exitCells
	exitCells = new int [numOfInitialActiveParticles];
	for (int i = 0; i < numOfInitialActiveParticles; i++)
		exitCells[i] = initialCellLocations[particleRecords[i]->GetGridPointID()];

	printf("before in device\n");

	// Initialize particle records in device
	InitializeParticleRecordsInDevice();
}

void lcsFastAdvection::InitializeVelocityData(double **velocities) {
	// Initialize velocity data
	for (int i = 0; i < 2; i++)
		velocities[i] = new double [globalNumOfPoints * 3];

	// Read velocities[0]
	frames[0]->GetTetrahedralGrid()->ReadVelocities(velocities[0]);
	
	// Create d_velocities[2]
	for (int i = 0; i < 2; i++) {
		err = cudaMalloc(&d_velocities[i], sizeof(double) * 3 * globalNumOfPoints);
		if (err) lcs::Error("Fail to create buffers for d_velocities[2]");
	}

	// Initialize d_velocities[0]
	err = cudaMemcpy(d_velocities[0], velocities[0], sizeof(double) * 3 * globalNumOfPoints, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to copy for d_velocities[0]");
}

void lcsFastAdvection::LoadVelocities(double *velocities, double *d_velocities, int frameIdx) {
	// Read velocities
	frames[frameIdx]->GetTetrahedralGrid()->ReadVelocities(velocities);

	/// DEBUG ///
/*
	FILE *fout = fopen("velocityTest.txt", "w");
	for (int i = 0; i < globalNumOfPoints; i++)
		fprintf(fout, "%.9lf %.9lf %.9lf\n", velocities[i * 3], velocities[i * 3 + 1], velocities[i * 3 + 2]);
	fclose(fout);
	exit(0);
*/	
	// Write for d_velocities[frameIdx]
	err = cudaMemcpy(d_velocities, velocities, sizeof(double) * 3 * globalNumOfPoints, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to copy for d_velocities");

#ifdef GET_VEL
	double average = 0;
	double largest = 0;
	for (int i = 0; i < globalNumOfPoints; i++) {
		double sum = 0;
		for (int j = 0; j < 3; j++)
			sum += velocities[i * 3 + j] * velocities[i * 3 + j];
		if (sum > largest) largest = sum;
		average += sqrt(sum);
	}
	average /= globalNumOfPoints;
	printf("Average velocity magnitude= %lf, Maximum velocity magnitude = %lf\n", average, sqrt(largest));
#endif
}

int lcsFastAdvection::CollectActiveParticlesForNewInterval(int *d_activeParticles) {
	// Prepare for exclusive scan
	InitializeScanArray(d_exitCells, d_exclusiveScanArrayForInt, numOfInitialActiveParticles);

	// Launch exclusive scan
	int sum;
	sum = ExclusiveScanForInt(d_exclusiveScanArrayForInt, numOfInitialActiveParticles);

	// Compaction
	CollectActiveParticles(d_exitCells, d_exclusiveScanArrayForInt, d_activeParticles, numOfInitialActiveParticles);

	// Return number of active particles
	return sum;
}

int lcsFastAdvection::CollectActiveParticlesForNewRun(int *d_oldActiveParticles, int *d_newActiveParticles, int length) {
	// Prepare for exclusive scan
	InitializeScanArray2(d_exitCells, d_oldActiveParticles, d_exclusiveScanArrayForInt, length);

	// Launch exclusive scan
	int sum;
	sum = ExclusiveScanForInt(d_exclusiveScanArrayForInt, length);

	/// DEBUG ///
	//printf("CollectActiveParticlesForNewRun(): length = %d, sum = %d\n", length, sum);

	// Compaction
	CollectActiveParticles2(d_exitCells, d_oldActiveParticles, d_exclusiveScanArrayForInt, d_newActiveParticles, length);

	// Return number of active particles
	return sum;
}

void lcsFastAdvection::InitializeInterestingBlockMarks() {
	err = cudaMalloc(&d_interestingBlockMarks, sizeof(int) * numOfInterestingBlocks);
	if (err) lcs::Error("Fail to create a buffer for device interestingBlockMarks");

	err = cudaMemset(d_interestingBlockMarks, 255, sizeof(int) * numOfInterestingBlocks);
	if (err) lcs::Error("Fail to initialize d_interestingBlockMarks");
}

int lcsFastAdvection::RedistributeParticles(int *d_activeParticles, int numOfActiveParticles, int iBMCount, int numOfStages) {
	/// DEBUG ///
	//err = cudaDeviceSynchronize();
	//printf("Before collect blocks Kernel, err = %d\n", err);

	/// DEBUG ///
	//printf("iBMCount = %d\n", iBMCount);

	// Intialize d_numOfActiveBlocks
	err = cudaMemset(d_numOfActiveBlocks, 0, sizeof(int));
	if (err) lcs::Error("Fail to initialize d_numOfActiveBlocks");

	// Launch collectActiveBlocksKernel
	CollectActiveBlocks(d_activeParticles, d_exitCells, d_placesOfInterest, d_localTetIDs, d_blockLocations, d_interestingBlockMap,
			d_startOffsetsInLocalIDMap, d_blocksOfTets, d_localIDsOfTets, d_interestingBlockMarks, d_activeBlocks,
			d_activeBlockIndices, d_numOfActiveBlocks, // Initially 0
			iBMCount, numOfActiveParticles, //int numOfStages,
			numOfBlocksInX, numOfBlocksInY, numOfBlocksInZ, globalMinX, globalMinY, globalMinZ,
			blockSize, this->GetEpsilon());

	// Get the number of active blocks
	int numOfActiveBlocks;

	err = cudaMemcpy(&numOfActiveBlocks, d_numOfActiveBlocks, sizeof(int), cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_numOfActiveBlocks");

	/// DEBUG ///
	//printf("numOfActiveBlocks = %d\n", numOfActiveBlocks);

	// Get the number of particles by stage in blocks
	err = cudaMemset(d_numOfParticlesByStageInBlocks, 0, numOfActiveBlocks * numOfStages * sizeof(int));
	if (err) lcs::Error("Fail to initialize d_numOfParticlesByStageInBlocks");

	GetNumOfParticlesByStageInBlocks(d_numOfParticlesByStageInBlocks, d_particleOrders, d_stages, d_activeParticles,
					 d_blockLocations, d_activeBlockIndices, numOfStages, numOfActiveParticles);

	// Prefix scan for d_numOfParticlesByStageInBlocks
	int sum;
	sum = ExclusiveScanForInt(d_numOfParticlesByStageInBlocks, numOfActiveBlocks * numOfStages);

	// Collect particles to blocks
	CollectParticlesToBlocks(d_numOfParticlesByStageInBlocks, // Now it is a prefix sum array.
				 d_particleOrders,
				 d_stages, d_activeParticles, d_blockLocations, d_activeBlockIndices, d_blockedActiveParticles,
				 numOfStages, numOfActiveParticles);

	// return
	return numOfActiveBlocks;
}

void lcsFastAdvection::GetStartOffsetInParticles(int numOfActiveBlocks, int numOfActiveParticles, int maxNumOfStages) {
	/// DEBUG ///
	//printf("In GetStartOffsetInParticles()\n");
	//printf("numOfActiveBlocks = %d\n", numOfActiveBlocks);
	//printf("numOfActiveParticles = %d\n", numOfActiveParticles);

	err = cudaMemcpy(d_startOffsetInParticles + numOfActiveBlocks, &numOfActiveParticles, sizeof(int), cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to d_startOffsetInParticles");

	CollectEveryKElement(d_numOfParticlesByStageInBlocks, d_startOffsetInParticles, maxNumOfStages, numOfActiveBlocks);
}

int lcsFastAdvection::AssignWorkGroups(int numOfActiveBlocks, int tracingBlockSize, int multiple) {
	// Get numOfGroupsForBlocks
	GetNumOfGroupsForBlocks(d_startOffsetInParticles, d_numOfGroupsForBlocks, numOfActiveBlocks, tracingBlockSize * multiple);

	// Exclusive scan of numOfGroupsForBlocks
	int sum;
	sum = ExclusiveScanForInt(d_numOfGroupsForBlocks, numOfActiveBlocks);

	// Fill in the sum
	err = cudaMemcpy(d_numOfGroupsForBlocks + numOfActiveBlocks, &sum, sizeof(int), cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to write to d_numOfGroupsForBlocks");

	// Assign groups
	AssignGroups(d_numOfGroupsForBlocks, // It should be the prefix sum now.
			d_blockOfGroups, d_offsetInBlocks, numOfActiveBlocks);

	return sum;
}

void lcsFastAdvection::CalculateBlockSizeAndSharedMemorySizeForTracingKernel(double averageParticlesInBlock, int &tracingBlockSize, int &tracingSharedMemorySize, int &multiple) {
	tracingBlockSize = (int)(averageParticlesInBlock / this->GetWarpSize()) * this->GetWarpSize();
	if (lcs::Sign(averageParticlesInBlock - tracingBlockSize, this->GetEpsilon()) > 0)
		tracingBlockSize += this->GetWarpSize();
	if (tracingBlockSize > this->GetMaxThreadsPerBlock())
		tracingBlockSize = this->GetMaxThreadsPerBlock();
/*
	for (int i = WARP_SIZE; ; i <<= 1)
		if (i > tracingBlockSize) {
			tracingBlockSize = i >> 1;
			break;
		}
*/
	if (tracingBlockSize < this->GetWarpSize())
		tracingBlockSize = this->GetWarpSize();

	/// DEBUG ///
	//tracingBlockSize = MAX_THREADS_PER_BLOCK;

	multiple = (int)(averageParticlesInBlock / tracingBlockSize);
	//multiple++;
	if (!multiple) multiple = 1;
	if (multiple > this->GetMaxMultiple()) multiple = this->GetMaxMultiple();

	int maxNumOfBlocks = this->GetMaxThreadsPerSM() / tracingBlockSize;

	tracingSharedMemorySize = this->GetMaxSharedMemoryPerSM() / maxNumOfBlocks;
	if (tracingSharedMemorySize > maxSharedMemoryRequired)
		tracingSharedMemorySize = maxSharedMemoryRequired;
	//printf("tracingBlockSize = %d, tracingSharedMemorySize = %d, multiple = %d\n", tracingBlockSize, tracingSharedMemorySize, multiple);
}

void lcsFastAdvection::Tracing() {
	printf("lcsFastAdvection::Tracing()\n");

	// Initialize d_tetrahedralLinks
	err = cudaMalloc(&d_tetrahedralLinks, sizeof(int) * globalNumOfCells * 4);
	if (err) lcs::Error("Fail to create a buffer for device tetrahedralLinks");

	err = cudaMemcpy(d_tetrahedralLinks, tetrahedralLinks, sizeof(int) * globalNumOfCells * 4, cudaMemcpyHostToDevice);
	if (err) lcs::Error("Fail to fill d_tetrahedralLinks");

	printf("before init active particles\n");

	// Initialize initial active particle data
	InitializeInitialActiveParticles();

	printf("after that\n");

	// Initialize velocity data
	double *velocities[2];
	int currStartVIndex = 1;
	InitializeVelocityData(velocities);

	printf("after init vel data\n");

	// Create some dynamic device arrays
	err = cudaMalloc(&d_exclusiveScanArrayForInt, sizeof(int) * std::max(numOfInterestingBlocks, numOfInitialActiveParticles));
	if (err) lcs::Error("Fail to create a buffer for device exclusiveScanArrayForInt");

	err = cudaMalloc(&d_blockOfGroups, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device blockOfGroups");

	err = cudaMalloc(&d_offsetInBlocks, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device offsetInBlocks");

	err = cudaMalloc(&d_numOfGroupsForBlocks, sizeof(int) * (numOfInterestingBlocks + 1));
	if (err) lcs::Error("Fail to create a buffer for device numOfGroupsForBlocks");

	err = cudaMalloc(&d_activeBlocks, sizeof(int) * numOfInterestingBlocks);
	if (err) lcs::Error("Fail to create a buffer for device activeBlocks");

	err = cudaMalloc(&d_activeBlockIndices, sizeof(int) * numOfInterestingBlocks);
	if (err) lcs::Error("Fail to create a buffer for device activeBlockIndices");

	err = cudaMalloc(&d_numOfActiveBlocks, sizeof(int));
	if (err) lcs::Error("Fail to create a buffer for device numOfActiveBlocks");

	err = cudaMalloc(&d_startOffsetInParticles, sizeof(int) * (numOfInterestingBlocks + 1));
	if (err) lcs::Error("Fail to create a buffer for device startOffsetInParticles");

	err = cudaMalloc(&d_blockedActiveParticles, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a buffer for device blockedAciveParticles");

	printf("before init interesting block marks\n");

	// Initialize interestingBlockMarks to {-1}
	InitializeInterestingBlockMarks();
	int iBMCount = 0;

	// Initialize numOfParticlesByStageInBlocks
	int maxNumOfStages;
	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: maxNumOfStages = 4; break;
	}

	err = cudaMalloc(&d_numOfParticlesByStageInBlocks, sizeof(int) * numOfInterestingBlocks * 4);
	if (err) lcs::Error("Fail to create a buffer for device numOfParticlesByStageInBlocks");

	// Initialize point positions in big blocks
	FABigBlockInitializationForPositions();

	// Some start setting
	currActiveParticleArray = 0;
	double currTime = 0;
	//double interval = configure->GetTimeInterval();
	
#ifdef TET_WALK_STAT
	err = cudaMalloc(&d_numOfTetWalks, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to create a device buffer for tetrahedral walk sums");

	err = cudaMemset(d_numOfTetWalks, 0, sizeof(int) * numOfInitialActiveParticles);
	if (err) lcs::Error("Fail to initialize d_numOfTetWalks");
#endif

	InitializeConstantsForBlockedTracingKernelOfRK4(d_vertexPositions, d_tetrahedralConnectivities,
				d_tetrahedralLinks, d_startOffsetInCell, d_startOffsetInPoint, d_vertexPositionsForBig, d_startVelocitiesForBig, d_endVelocitiesForBig, 
				d_localConnectivities, d_localLinks, d_globalCellIDs, d_activeBlocks, // Map active block ID to interesting block ID
				d_blockOfGroups, d_offsetInBlocks, d_stages, d_lastPositionForRK4,
				d_kForRK4, d_nxForRK4, /*d_k2ForRK4, d_k3ForRK4,*/ d_pastTimes, d_placesOfInterest,
				d_startOffsetInParticles, d_blockedActiveParticles, d_localTetIDs, d_exitCells, this->caller->GetTimeStep(), this->GetEpsilon()

#ifdef TET_WALK_STAT
				, d_numOfTetWalks
#endif

);
	
	// Main loop for blocked tracing
	/// DEBUG ///
	//int startTime = clock();

	//numOfTimePoints = this->numOfTimePoints; ->GetNumOfTimePoints();
	double startTime = lcs::GetCurrentTimeInSeconds();

	/// DEBUG ///
	kernelSum = 0;
	int numOfKernelCalls = 0;

#ifdef BLOCK_STAT
	FILE *blockStatFile1 = fopen("lcsBlockStat1.txt", "w");
	fprintf(blockStatFile1, "No. of call\tNo. of active particles\tNo. of active blocks\tAve. of particles\n");
	FILE *blockStatFile2 = fopen("lcsBlockStat2.txt", "w");
	fprintf(blockStatFile2, "No. of call %6s", "lower");
	int lowerBound = 512, upperBound = 1024;
	for (int i = lowerBound; i <= upperBound; i++)
		if (i % WARP_SIZE == 0) fprintf(blockStatFile2, " %6d", i);
	fprintf(blockStatFile2, " %6s\n", "upper");
	int *startOffsetInActiveParticles = new int [numOfInitialActiveParticles];
	int *histogram = new int [upperBound + 1];
#endif

	double interval;

	printf("begin main loop\n");

	for (int frameIdx = 0; currTime < this->caller->GetAdvectionTime(); frameIdx++, currTime += interval) {
		interval = frameIdx + 1 < this->numOfFrames ? this->input->GetTimePoint(frameIdx + 1) - this->input->GetTimePoint(frameIdx) :
			                                          this->input->GetTimePoint(1) - this->input->GetTimePoint(0);
		if (frameIdx == this->numOfFrames) frameIdx = 0;
		interval = std::min(interval, this->caller->GetAdvectionTime() - currTime);

//#ifdef BLOCK_STAT
//		if (frameIdx) {
//			fprintf(blockStatFile1, "\n");
//			fprintf(blockStatFile2, "\n");
//		}
//#endif

//#ifdef GET_PATH
//		char fileName[100];
//		sprintf(fileName, "lcsPositions%02d.vtk", frameIdx);
//		GetLastPositions(fileName, currTime);
//#endif

		printf("*********Tracing between time point %d and time point %d*********\n", frameIdx, frameIdx + 1);
		printf("\n");

		/// DEBUG ///
		int startTime;
		//startTime = clock();

		currStartVIndex = 1 - currStartVIndex;

		// Collect active particles
		int lastNumOfActiveParticles;

		lastNumOfActiveParticles = CollectActiveParticlesForNewInterval(d_activeParticles[currActiveParticleArray]);

		/// DEBUG ///
		printf("numOfActiveParticles = %d\n", lastNumOfActiveParticles);

		// Load end velocities
		LoadVelocities(velocities[1 - currStartVIndex], d_velocities[1 - currStartVIndex], (frameIdx + 1) % numOfFrames);

		// Initialize big blocks
		FABigBlockInitializationForVelocities(currStartVIndex);

		/// DEBUG ///
		//printf("BigBlockInitializationForVelocities done.\n");

		startTime = clock();
		kernelSumInInterval = 0;

		while (true) {
			// Get active particles
			currActiveParticleArray = 1 - currActiveParticleArray;

			int numOfActiveParticles;

			numOfActiveParticles = CollectActiveParticlesForNewRun(d_activeParticles[1 - currActiveParticleArray],
									       d_activeParticles[currActiveParticleArray],
									       lastNumOfActiveParticles);

			lastNumOfActiveParticles = numOfActiveParticles;

			if (!numOfActiveParticles) break;

			/// DEBUG ///
			numOfKernelCalls++;

			int numOfActiveBlocks = RedistributeParticles(d_activeParticles[currActiveParticleArray],
								      numOfActiveParticles, iBMCount++, maxNumOfStages);	

			double averageParticlesInBlock = (double)numOfActiveParticles / numOfActiveBlocks;
			//printf("numOfActiveParticles / numOfActiveBlocks = %lf\n", averageParticlesInBlock);

			GetStartOffsetInParticles(numOfActiveBlocks, numOfActiveParticles, maxNumOfStages);

#ifdef BLOCK_STAT
			fprintf(blockStatFile1, "%11d\t%23d\t%20d\t%17.2lf\n", numOfKernelCalls, numOfActiveParticles, numOfActiveBlocks, averageParticlesInBlock);
			err = cudaMemcpy(startOffsetInActiveParticles, d_startOffsetInParticles, sizeof(int) * (numOfActiveBlocks + 1), cudaMemcpyDeviceToHost);
			if (err) lcs::Error("Fail to read d_startOffsetInActiveParticles");
			memset(histogram, 0, sizeof(int) * (upperBound + 1));
			int lowerSum = 0, upperSum = 0;
			for (int i = 0; i < numOfActiveBlocks; i++) {
				int pop = startOffsetInActiveParticles[i + 1] - startOffsetInActiveParticles[i];
				if (pop < lowerBound) lowerSum++;
				else if (pop > upperBound) upperSum++;
				else histogram[pop / WARP_SIZE * WARP_SIZE]++;
			}
			fprintf(blockStatFile2, "%11d %6d", numOfKernelCalls, lowerSum);
			for (int i = lowerBound; i <= upperBound; i++)
				if (i % WARP_SIZE == 0) fprintf(blockStatFile2, " %6d", histogram[i]);
			fprintf(blockStatFile2, " %6d\n", upperSum);
#endif

			int tracingBlockSize, tracingSharedMemorySize, multiple;
			CalculateBlockSizeAndSharedMemorySizeForTracingKernel(averageParticlesInBlock, tracingBlockSize, tracingSharedMemorySize, multiple);

			int numOfWorkGroups = AssignWorkGroups(numOfActiveBlocks, tracingBlockSize, multiple);

			LaunchBlockedTracingKernel(numOfWorkGroups, currTime, currTime + interval, tracingBlockSize, tracingSharedMemorySize, multiple);
		}

		int endTime = clock();
		printf("This interval cost %lf sec.\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
		printf("kernelSumInInterval = %lf sec.\n", kernelSumInInterval);
		printf("\n");
	}

#ifdef BLOCK_STAT
	fclose(blockStatFile1);
	fclose(blockStatFile2);
	delete [] startOffsetInActiveParticles;
	delete [] histogram;
#endif

	// Release device resources
	cudaFree(d_exclusiveScanArrayForInt);

	/// DEBUG ///
	printf("kernelSum = %lf\n", kernelSum);
	printf("numOfKernelCalls = %d\n", numOfKernelCalls);

	/// DEBUG ///
	double endTime = lcs::GetCurrentTimeInSeconds();
	//int endTime = clock();
	printf("The total tracing time is %lf sec.\n", endTime - startTime);//(double)(endTime - startTime) / CLOCKS_PER_SEC);
	printf("\n");

#ifdef TET_WALK_STAT
	int *numOfTetWalks = new int [numOfInitialActiveParticles];
	err = cudaMemcpy(numOfTetWalks, d_numOfTetWalks, sizeof(int) * numOfInitialActiveParticles, cudaMemcpyDeviceToHost);
	if (err) lcs::Error("Fail to read d_numOfTetWalks");
	double totalTetWalks = 0;
	int  minTetWalks = numOfTetWalks[0], maxTetWalks = numOfTetWalks[0];
	for (int i = 0; i < numOfInitialActiveParticles; i++) {
		totalTetWalks += numOfTetWalks[i];
		if (numOfTetWalks[i] < minTetWalks) minTetWalks = numOfTetWalks[i];
		if (numOfTetWalks[i] > maxTetWalks) maxTetWalks = numOfTetWalks[i];
	}
	printf("Average number of tetrahedron walk is %lf.\n", (double)totalTetWalks / numOfInitialActiveParticles);
	printf("minTetWalks = %d, maxTetWalks = %d\n", minTetWalks, maxTetWalks);
	printf("\n");

	delete [] numOfTetWalks;
#endif

	for (int i = 0; i < 2; i++)
		delete [] velocities[i];
}

void lcsFastAdvection::GetLastPositions(vtkRectilinearGrid *finalOutput) {
	double range[2];

	this->caller->GetXRange(range);
	double minX = range[0];
	double maxX = range[1];

	this->caller->GetYRange(range);
	double minY = range[0];
	double maxY = range[1];

	this->caller->GetZRange(range);
	double minZ = range[0];
	double maxZ = range[1];

	int dimensions[3];

	this->caller->GetDimensions(dimensions);
	int xRes = dimensions[0];
	int yRes = dimensions[1];
	int zRes = dimensions[2];

	double dx = (maxX - minX) / xRes;
	double dy = (maxY - minY) / yRes;
	double dz = (maxZ - minZ) / zRes;

	vtkSmartPointer<vtkRectilinearGrid> output = vtkSmartPointer<vtkRectilinearGrid>::New();

	output->SetExtent(0, dimensions[0], 0, dimensions[1], 0, dimensions[2]);
	
	vtkDoubleArray *xCoords = vtkDoubleArray::New();
	xCoords->SetNumberOfComponents(1);
	xCoords->SetNumberOfTuples(dimensions[0] + 1);
	for (int i = 0; i <= dimensions[0]; i++)
		xCoords->SetTuple1(i, minX + dx * i);
	output->SetXCoordinates(xCoords);

	vtkDoubleArray *yCoords = vtkDoubleArray::New();
	yCoords->SetNumberOfComponents(1);
	yCoords->SetNumberOfTuples(dimensions[1] + 1);
	for (int i = 0; i <= dimensions[1]; i++)
		yCoords->SetTuple1(i, minY + dy * i);
	output->SetYCoordinates(yCoords);

	vtkDoubleArray *zCoords = vtkDoubleArray::New();
	zCoords->SetNumberOfComponents(1);
	zCoords->SetNumberOfTuples(dimensions[2] + 1);
	for (int i = 0; i <= dimensions[2]; i++)
		zCoords->SetTuple1(i, minZ + dz * i);
	output->SetZCoordinates(zCoords);

	vtkDoubleArray *dest = vtkDoubleArray::New();
	dest->SetNumberOfComponents(3);
	dest->SetNumberOfTuples(output->GetNumberOfPoints());

	//xCoords->Delete();
	//yCoords->Delete();
	//zCoords->Delete();

	/// DEBUG ///
	printf("dest->GetNumberOfTuples() = %d\n", dest->GetNumberOfTuples());

	for (int i = 0; i <= xRes; i++)
		for (int j = 0; j <= yRes; j++)
			for (int k = 0; k <= zRes; k++) {
				int ijk[3] = {i, j, k};
				int pointId = output->ComputePointId(ijk);
				dest->SetTuple3(pointId, minX + dx * i, minY + dy * j, minZ + dz * k);
			}

	output->GetPointData()->SetVectors(dest);
	
	this->finalPositions = new double [numOfInitialActiveParticles * 3];

	switch (lcs::ParticleRecord::GetDataType()) {
	case lcs::ParticleRecord::RK4: {
		cudaMemcpy(this->finalPositions, d_lastPositionForRK4, sizeof(double) * 3 * numOfInitialActiveParticles, cudaMemcpyDeviceToHost);
	} break;
	}
	for (int i = 0; i < numOfInitialActiveParticles; i++) {
		int gridPointID = particleRecords[i]->GetGridPointID();
		int z = gridPointID % (zRes + 1);
		int temp = gridPointID / (zRes + 1);
		int y = temp % (yRes + 1);
		int x = temp / (yRes + 1);

		int ijk[3] = {x, y, z};
		int pointId = output->ComputePointId(ijk);
		dest->SetTuple(pointId, finalPositions + i * 3);
	}

	//dest->Delete();

	/// DEBUG ///
	
	//vtkSmartPointer<vtkXMLRectilinearGridWriter> writer = vtkSmartPointer<vtkXMLRectilinearGridWriter>::New();
	//writer->SetFileName("test.vtu");

	//vtkSmartPointer<vtkRectilinearGrid> newGrid = vtkSmartPointer<vtkRectilinearGrid>::New();
	//newGrid->CopyStructure(output);
	//newGrid->CopyAttributes(output);

	//writer->SetInput(newGrid);
	//writer->Write();
	
	finalOutput->CopyStructure(output);
	finalOutput->CopyAttributes(output);

	delete [] this->finalPositions;
}

void lcsFastAdvection::ComputeFlowMap(lcsAxisAlignedFlowMap *caller, lcsUnstructuredGridWithTimeVaryingPointData *input, vtkRectilinearGrid *output) {
	printf("ComputeFlowMap\n");

	this->caller = caller;
	this->input = input;

	if (this->GetIntegrationMethod() == RK4) lcs::ParticleRecord::SetDataType(lcs::ParticleRecord::RK4);

	// Load all the frames
	this->LoadFrames(input);
	
	// Put both topological and geometrical data into arrays
	this->GetTopologyAndGeometry();
	
	// Get the global bounding box
	this->GetGlobalBoundingBox();
	
	// Calculate the number of blocks in X, Y and Z
	this->CalculateNumOfBlocksInXYZ();
	
	// Divide the flow domain into blocks
	this->Division();

	// Initially locate global tetrahedral cells for interesting Cartesian grid points
	this->AAInitialCellLocation();

	// Main Tracing Process
	this->Tracing();

	// Get last positions
	this->GetLastPositions(output);

	this->CleanAfterFlowMapComputation();
}