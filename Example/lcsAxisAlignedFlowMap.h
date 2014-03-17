// .NAME lcsAxisAlignedFlowMap
// .SECTION Description
// lcsAxisAlignedFlowMap is a filter which computes the flow map of
// an axis-aligned sample grid over an unstructured grid with time-
// varying velocity data.

#ifndef __lcsAxisAlignedFlowMap_h
#define __lcsAxisAlignedFlowMap_h

#include <vtkRectilinearGridAlgorithm.h>

#define SET_RANGE(dim) \
	void Set##dim##Range(double lower, double upper) { \
		this->dim##Range[0] = lower; \
		this->dim##Range[1] = upper; \
		this->Modified(); \
	} \
	void Set##dim##Range(double *range) { \
		this->Set##dim##Range(range[0], range[1]); \
		this->Modified(); \
	} \
	void Get##dim##Range(double *range) { \
		range[0] = this->dim##Range[0]; \
		range[1] = this->dim##Range[1]; \
	}

class lcsAxisAlignedFlowMap : public vtkRectilinearGridAlgorithm {
public:
	static lcsAxisAlignedFlowMap *New();
	vtkTypeMacro(lcsAxisAlignedFlowMap, vtkRectilinearGridAlgorithm);
	vtkSetMacro(AdvectionTime, double);
	vtkGetMacro(AdvectionTime, double);
	vtkSetMacro(TimeStep, double);
	vtkGetMacro(TimeStep, double);

	vtkGetMacro(BlockSize, double);
	vtkGetMacro(MarginRatio, double);

	void PrintSelf(ostream &os, vtkIndent indent);

	SET_RANGE(X);
	SET_RANGE(Y);
	SET_RANGE(Z);
	
	void SetDimensions(int nx, int ny, int nz) {
		this->Dimensions[0] = nx;
		this->Dimensions[1] = ny;
		this->Dimensions[2] = nz;

		this->Modified();
	}

	void GetDimensions(int *dimensions) {
		dimensions[0] = this->Dimensions[0];
		dimensions[1] = this->Dimensions[1];
		dimensions[2] = this->Dimensions[2];
	}

	void UseDefaultSetting() {
		this->DefaultSetting = true; // It is just about the performance, not about the result, thus does not need to call Modified().
	}

	bool DefaultSettingOn() {
		return this->DefaultSetting;
	}

	void SetBlockSizeAndMarginRatio(double blockSize, double marginRatio) {
		this->BlockSize = blockSize;
		this->MarginRatio = marginRatio;
		this->DefaultSetting = false;
	}

protected:
	lcsAxisAlignedFlowMap();
	~lcsAxisAlignedFlowMap();
	
	virtual int FillInputPortInformation(int, vtkInformation *);
	virtual int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *outputVector);
	virtual int RequestData(vtkInformation *request, vtkInformationVector **inputVector, vtkInformationVector *outputVector);

	double XRange[2], YRange[2], ZRange[2];
	int Dimensions[3];
	double AdvectionTime;
	double TimeStep;
	double BlockSize;
	double MarginRatio;

	bool DefaultSetting;

private:
	lcsAxisAlignedFlowMap(const lcsAxisAlignedFlowMap &); // Not implemented.
	void operator = (const lcsAxisAlignedFlowMap &); // Not implemented.
};

#endif