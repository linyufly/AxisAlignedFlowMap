#include "lcsAxisAlignedFlowMap.h"
#include "lcsUnstructuredGridWithTimeVaryingPointData.h"
#include "lcsFastAdvection.h"

#include <vtkObjectFactory.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkRectilinearGrid.h>

vtkStandardNewMacro(lcsAxisAlignedFlowMap);

lcsAxisAlignedFlowMap::lcsAxisAlignedFlowMap() {
	this->XRange[0] = 0.0;
	this->XRange[1] = 1.0;

	this->YRange[0] = 0.0;
	this->YRange[1] = 1.0;

	this->ZRange[0] = 0.0;
	this->ZRange[1] = 1.0;

	this->Dimensions[0] = 2;
	this->Dimensions[1] = 2;
	this->Dimensions[2] = 2;
}

lcsAxisAlignedFlowMap::~lcsAxisAlignedFlowMap() {
}

void lcsAxisAlignedFlowMap::PrintSelf(ostream &os, vtkIndent indent) {
	this->Superclass::PrintSelf(os, indent);

	os << indent << "XRange: " << this->XRange[0] << ", " << this->XRange[1] << "\n";
	os << indent << "YRange: " << this->YRange[0] << ", " << this->YRange[1] << "\n";
	os << indent << "ZRange: " << this->ZRange[0] << ", " << this->ZRange[1] << "\n";

	os << indent << "Dimensions: " << this->Dimensions[0] << ", " << this->Dimensions[1] << ", " << this->Dimensions[2] << "\n";
}

int lcsAxisAlignedFlowMap::FillInputPortInformation(int port, vtkInformation *info) {
	printf("FillInputPortInformation\n");

	if(!this->Superclass::FillInputPortInformation(port, info))
		return 0;

	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "lcsUnstructuredGridWithTimeVaryingPointData");

	return 1;
}

int lcsAxisAlignedFlowMap::RequestData(vtkInformation *request, vtkInformationVector **inputVector, vtkInformationVector *outputVector) {
	printf("RequestData\n");

	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	lcsUnstructuredGridWithTimeVaryingPointData *input = lcsUnstructuredGridWithTimeVaryingPointData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
	if (!input)
		return 0;

	vtkInformation *outInfo = outputVector->GetInformationObject(0);
	vtkRectilinearGrid *output = vtkRectilinearGrid::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
	if (!output)
		return 0;

	vtkDebugMacro(<< "Executing axis-aligned flow map filter");

	lcsFastAdvection advector;
	advector.ComputeFlowMap(this, input, output);

	return 1;
}