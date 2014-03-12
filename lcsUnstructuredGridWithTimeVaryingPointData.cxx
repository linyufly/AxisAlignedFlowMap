#include "lcsUnstructuredGridWithTimeVaryingPointData.h"

#include <vtkObjectFactory.h>
#include <vtkPointData.h>

vtkStandardNewMacro(lcsUnstructuredGridWithTimeVaryingPointData);

lcsUnstructuredGridWithTimeVaryingPointData::lcsUnstructuredGridWithTimeVaryingPointData() {
	this->NumberOfSamples = 0;
	this->TimePoints = NULL;
	this->TimeVaryingPointDataArray = NULL;
}

lcsUnstructuredGridWithTimeVaryingPointData::~lcsUnstructuredGridWithTimeVaryingPointData() {
	if (this->TimePoints)
		delete [] this->TimePoints;

	if (this->NumberOfSamples > 0) {
		for (int i = 0; i < this->NumberOfSamples; i++)
			this->TimeVaryingPointDataArray[i]->Delete();
		delete [] this->TimeVaryingPointDataArray;
	}
}

void lcsUnstructuredGridWithTimeVaryingPointData::PrintSelf(ostream &os, vtkIndent indent) {
	this->Superclass::PrintSelf(os, indent);
	
	os << indent << "Number Of Samples: " << this->GetNumberOfSamples() << endl;
}

vtkPointData *lcsUnstructuredGridWithTimeVaryingPointData::GetTimeVaryingPointData(vtkIdType timePointId) {
	if (this->TimeVaryingPointDataArray[timePointId] == NULL)
		this->TimeVaryingPointDataArray[timePointId] = vtkPointData::New();
	return this->TimeVaryingPointDataArray[timePointId];
}

void lcsUnstructuredGridWithTimeVaryingPointData::SetNumberOfSamples(int numberOfSamples) {
	if (this->TimeVaryingPointDataArray) {
		for (int i = 0; i < this->NumberOfSamples; i++)
			if (this->TimeVaryingPointDataArray[i]) {
				this->TimeVaryingPointDataArray[i]->Delete();
				this->TimeVaryingPointDataArray[i] = NULL;
			}
	}

	if (numberOfSamples > this->NumberOfSamples) {
		delete [] this->TimeVaryingPointDataArray;
		this->TimeVaryingPointDataArray = new vtkPointData *[numberOfSamples];
		for (int i = 0; i < numberOfSamples; i++)
			this->TimeVaryingPointDataArray[i] = NULL;
	}

	this->NumberOfSamples = numberOfSamples;
}

unsigned long lcsUnstructuredGridWithTimeVaryingPointData::GetMTime() { // Follow the code of vtkPointSet::GetMTime() and vtkDataSet::GetMTime().
	unsigned long result = this->Superclass::GetMTime();

	if (this->HasTimePoints()) {
		for (int i = 0; i < this->NumberOfSamples; i++) {
			unsigned long temp = this->GetTimeVaryingPointData(i)->GetMTime();
			if (temp > result) result = temp;
		}
	}

	return result;
}