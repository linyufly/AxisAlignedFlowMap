#ifndef __FastAdvection_h
#define __FastAdvection_h

#include "lcs.h"
#include "lcsGeometry.h"

#include <cuda_runtime.h>

class lcsUnstructuredGridWithTimeVaryingPointData;
class lcsAxisAlignedFlowMap;
class vtkRectilinearGrid;

#define lcsGetMacro(name, type) \
	type Get##name() const { \
		return this->name; \
	}

#define lcsSetMacro(name, type) \
	void Set##name(type value) { \
		this->name = value; \
	}

class lcsFastAdvection {
public:
	enum IntegrationType {RK4};

	lcsFastAdvection::lcsFastAdvection() {
		// Up to user
		this->IntegrationMethod = RK4;
		this->UseUnitTestForTetBlkIntersection = false;
		this->UseUnitTestForInitialCellLocation = true;

		// Up to data
		this->BlockSize = 0.15; //1.0;
		this->MarginRatio = 0.3;

		// Up to data, but seldom requires changes
		this->EpsilonForTetBlkIntersection = 1e-10;
		this->Epsilon = 1e-8;

		// Up to system
		this->MaxThreadsPerSM = 512;
		this->MaxThreadsPerBlock = 256;
		this->MaxSharedMemoryPerSM = 49000; //49152
		this->WarpSize = 32;
		this->MaxMultiple = 16;
	}

	lcsGetMacro(BlockSize, double);
	lcsGetMacro(MarginRatio, double);
	lcsGetMacro(EpsilonForTetBlkIntersection, double);
	lcsGetMacro(Epsilon, double);
	lcsGetMacro(IntegrationMethod, IntegrationType);
	lcsGetMacro(MaxThreadsPerSM, int);
	lcsGetMacro(MaxThreadsPerBlock, int);
	lcsGetMacro(MaxSharedMemoryPerSM, int);
	lcsGetMacro(WarpSize, int);
	lcsGetMacro(MaxMultiple, int);
	lcsGetMacro(UseUnitTestForTetBlkIntersection, bool);
	lcsGetMacro(UseUnitTestForInitialCellLocation, bool);

	lcsSetMacro(BlockSize, double);
	lcsSetMacro(MarginRatio, double);
	lcsSetMacro(EpsilonForTetBlkIntersection, double);
	lcsSetMacro(Epsilon, double);
	lcsSetMacro(IntegrationMethod, IntegrationType);
	lcsSetMacro(MaxThreadsPerSM, int);
	lcsSetMacro(MaxThreadsPerBlock, int);
	lcsSetMacro(MaxSharedMemoryPerSM, int);
	lcsSetMacro(WarpSize, int);
	lcsSetMacro(MaxMultiple, int);
	lcsSetMacro(UseUnitTestForTetBlkIntersection, bool);
	lcsSetMacro(UseUnitTestForInitialCellLocation, bool);

	// If the advection time is beyond the last time points, the last time point is assumed equal to the
	// first time point and we do periodic advection.
	void ComputeFlowMap(lcsAxisAlignedFlowMap *caller, lcsUnstructuredGridWithTimeVaryingPointData *input, vtkRectilinearGrid *output);

private:
	double BlockSize;
	double MarginRatio;
	double EpsilonForTetBlkIntersection;
	double Epsilon;

	IntegrationType IntegrationMethod;

	int MaxThreadsPerSM;
	int MaxThreadsPerBlock;
	int MaxSharedMemoryPerSM;
	int WarpSize;
	int MaxMultiple;

	bool UseUnitTestForTetBlkIntersection;
	bool UseUnitTestForInitialCellLocation;

	void CleanAfterFlowMapComputation();

	lcsAxisAlignedFlowMap *caller;
	lcsUnstructuredGridWithTimeVaryingPointData *input;

	// Modified from original Fastvection project. The following methods serve the flow map computation.
	void LoadFrames(lcsUnstructuredGridWithTimeVaryingPointData *grid);
	int GetBlockID(int x, int y, int z);
	void GetXYZFromBlockID(int blockID, int &x, int &y, int &z);
	void GetXYZFromPosition(const lcs::Vector &position, int &x, int &y, int &z);
	void GetTopologyAndGeometry();
	void GetGlobalBoundingBox();
	void CalculateNumOfBlocksInXYZ();
	void PrepareTetrahedronBlockIntersectionQueries();
	void LaunchGPUforIntersectionQueries();
	void DivisionProcess();
	void StoreBlocksInDevice();
	void Division();
	void AAInitialCellLocation();
	void InitializeParticleRecordsInDevice();
	void FABigBlockInitializationForPositions();
	void FABigBlockInitializationForVelocities(int currStartVIndex);
	void LaunchBlockedTracingKernel(int numOfWorkGroups, double beginTime, double finishTime, int blockSize, int sharedMemorySize, int multiple);
	void InitializeInitialActiveParticles();
	void InitializeVelocityData(double **velocities);
	void LoadVelocities(double *velocities, double *d_velocities, int frameIdx);
	int CollectActiveParticlesForNewInterval(int *d_activeParticles);
	int CollectActiveParticlesForNewRun(int *d_oldActiveParticles, int *d_newActiveParticles, int length);
	void InitializeInterestingBlockMarks();
	int RedistributeParticles(int *d_activeParticles, int numOfActiveParticles, int iBMCount, int numOfStages);
	void GetStartOffsetInParticles(int numOfActiveBlocks, int numOfActiveParticles, int maxNumOfStages);
	int AssignWorkGroups(int numOfActiveBlocks, int tracingBlockSize, int multiple);
	void CalculateBlockSizeAndSharedMemorySizeForTracingKernel(double averageParticlesInBlock, int &tracingBlockSize, int &tracingSharedMemorySize, int &multiple);
	void Tracing();
	void GetLastPositions(vtkRectilinearGrid *output);

	// Modified from original Fastvection project. The following variables will be created once for computing
	// flow map, and then deleted afterwards.
	lcs::Frame **frames;
	int numOfFrames;
	int numOfTimePoints;

	int *tetrahedralConnectivities, *tetrahedralLinks;
	double *vertexPositions;
	int globalNumOfCells, globalNumOfPoints;
	double globalMinX, globalMaxX, globalMinY, globalMaxY, globalMinZ, globalMaxZ;

	double blockSize;
	int numOfBlocksInX, numOfBlocksInY, numOfBlocksInZ;

	// For FTLE calculation
	double *finalPositions;

	// For tetrahedron-block intersection
	int *xLeftBound, *xRightBound, *yLeftBound, *yRightBound, *zLeftBound, *zRightBound;
	int numOfQueries;
	int *queryTetrahedron, *queryBlock;
	char *queryResults; // Whether certain tetrahedron intersects with certain block

	// For blocks
	int numOfBlocks, numOfInterestingBlocks;
	lcs::BlockRecord **blocks;
	int *startOffsetInCell, *startOffsetInPoint;

	// For initial cell location
	int *initialCellLocations;

	// For tracing
	lcs::ParticleRecord **particleRecords;
	int *exitCells;
	int numOfInitialActiveParticles;

	// For shared memory
	int maxSharedMemoryRequired;

	/// DEBUG ///
	double kernelSum;
	double kernelSumInInterval;

	// CUDA C variables

	// error
	cudaError_t err;

	#ifdef TET_WALK_STAT
	// Device memory for tetrahedron walk statistics
	int *d_numOfTetWalks;
	#endif

	// Device memory for exclusive scan for int
	int *d_exclusiveScanArrayForInt;

	// Device memory for interesting block map
	int *d_interestingBlockMap;

	// Device memory for (tet, blk) to local tet ID map
	int *d_startOffsetsInLocalIDMap;
	int *d_blocksOfTets;
	int *d_localIDsOfTets;

	// Device memory for particle redistribution
	int *d_numOfParticlesByStageInBlocks; // It depends on the maximum stage number of the integration method.
	int *d_interestingBlockMarks;
	int *d_particleOrders; // The local order number in (block, stage) group
	int *d_blockLocations;

	// Device memory for global geometry
	int *d_tetrahedralConnectivities, *d_tetrahedralLinks;
	double *d_vertexPositions;
	int *d_queryTetrahedron, *d_queryBlock;
	bool *d_queryResults;

	// Device memory for cell locations of particles
	int *d_cellLocations;

	// Device memory for local geometry in blocks
	int *d_localConnectivities, *d_localLinks;
	int *d_globalCellIDs, *d_globalPointIDs;
	int *d_startOffsetInCell, *d_startOffsetInPoint;

	// Device memory for particle
	int *d_activeBlockOfParticles;
	int *d_localTetIDs;
	int *d_exitCells;
	int *d_activeParticles[2];

	int currActiveParticleArray;

	int *d_stages;
	double *d_lastPositionForRK4;

	double *d_kForRK4, *d_nxForRK4;
	double *d_pastTimes;
	double *d_placesOfInterest;

	// Device memory for velocities
	double *d_velocities[2];

	// Device memory for big blocks
	double *d_vertexPositionsForBig, *d_startVelocitiesForBig, *d_endVelocitiesForBig;

	// Device memory for canFitInSharedMemory flags
	//bool *d_canFitInSharedMemory;

	// Device memory for active block list
	int *d_activeBlocks;
	int *d_activeBlockIndices;
	int *d_numOfActiveBlocks;

	// Device memory for tracing work groups distribution
	int *d_numOfGroupsForBlocks;
	int *d_blockOfGroups;
	int *d_offsetInBlocks;

	// Device memory for start offsets of particles in active blocks
	int *d_startOffsetInParticles;

	// Device memory for particles grouped in blocks
	int *d_blockedActiveParticles;
};

#endif