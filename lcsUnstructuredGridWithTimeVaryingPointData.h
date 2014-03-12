// .NAME lcsUnstructuredGridWithTimeVaryingPointData
// .SECTION Description
// lcsUnstructuredGridWithTimeVaryingPointData is an ad-hoc class for
// Lagrangian Coherent Structures (LCS) computation. It is suitable for
// the cases where the geometry and topology of the flow domain are
// fixed but each point has a series of sampled data along with time.
// The time point sequence for all the points are the same. Compared to
// a series of vtkUnstructuredGrid, we avoid the duplication of geometry
// and topology information.
//
// We are looking forward to better support for time-varying data in VTK
// package, and we will update our tool based on that.

#ifndef __lcsUnstructuredGridWithTimeVaryingPointData_h
#define __lcsUnstructuredGridWithTimeVaryingPointData_h

#include <vtkUnstructuredGrid.h>

class vtkPointData;

class lcsUnstructuredGridWithTimeVaryingPointData : public vtkUnstructuredGrid {
public:
	static lcsUnstructuredGridWithTimeVaryingPointData *New();
	vtkTypeMacro(lcsUnstructuredGridWithTimeVaryingPointData, vtkUnstructuredGrid);

	void PrintSelf(ostream &os, vtkIndent indent);
	unsigned long int GetMTime();

	vtkGetMacro(NumberOfSamples, int);

	void SetNumberOfSamples(int numberOfSamples);

	void SetUnstructuredGrid(vtkUnstructuredGrid *grid) {
		this->CopyStructure(grid);
	}

	bool HasTimePoints() const {
		return this->TimePoints == NULL;
	}

	void SetTimePoint(vtkIdType id, double timePoint) {
		this->TimePoints[id] = timePoint;

		this->Modified();
	}

	double GetTimePoint(vtkIdType id) const {
		return this->TimePoints[id];
	}

	void SetUniformTimePoints(double startTime, double endTime) {
		if (this->TimePoints == NULL)
			delete [] this->TimePoints;

		this->TimePoints = new double [this->NumberOfSamples];

		for (int i = 0; i < this->NumberOfSamples; i++)
			this->TimePoints[i] = i * (endTime - startTime) / (this->NumberOfSamples - 1) + startTime;

		this->Modified();
	}

	void SetTimePoints(double *timePoints) {
		for (int i = 0; i < this->NumberOfSamples; i++)
			this->TimePoints[i] = timePoints[i];

		this->Modified();
	}

	void GetTimePoints(double *timePoints) {
		for (int i = 0; i < this->NumberOfSamples; i++)
			timePoints[i] = this->TimePoints[i];
	}

	vtkPointData *GetTimeVaryingPointData(vtkIdType timePointId);

protected:
	lcsUnstructuredGridWithTimeVaryingPointData();
	~lcsUnstructuredGridWithTimeVaryingPointData();

	double *TimePoints;
	int NumberOfSamples;
	vtkPointData **TimeVaryingPointDataArray;

private:
	lcsUnstructuredGridWithTimeVaryingPointData(const lcsUnstructuredGridWithTimeVaryingPointData &); // Not implemented.
	void operator = (const lcsUnstructuredGridWithTimeVaryingPointData &); // Not implemented.
};

#endif

