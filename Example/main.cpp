#include "lcsAxisAlignedFlowMap.h"
#include "lcsUnstructuredGridWithTimeVaryingPointData.h"

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkPointData.h>
#include <vtkRectilinearGridWriter.h>
#include <vtkXMLRectilinearGridWriter.h>

#include <cstdio>

#include <string>

#define NUM_OF_SAMPLES 2

std::string ToStr(int num) {
	static char str[100];
	sprintf(str, "%d", num);
	return std::string(str);
}

int main() {
	lcsUnstructuredGridWithTimeVaryingPointData *grid = lcsUnstructuredGridWithTimeVaryingPointData::New();
	grid->SetNumberOfSamples(NUM_OF_SAMPLES);
	grid->SetUniformTimePoints(0, (NUM_OF_SAMPLES - 1) * 0.04);
	for (int i = 3040; i <= 4040; i += 40) {
		if ((i - 3040) / 40 >= NUM_OF_SAMPLES) break;
		printf("i = %d\n", i);
		std::string name = "./vtkfiles/Patient2Rest_vel." + ToStr(i) + ".vtk";
		vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
		reader->SetFileName(name.c_str());
		reader->Update();
		if (i == 3040) grid->SetUnstructuredGrid(reader->GetOutput());
		grid->GetTimeVaryingPointData((i - 3040) / 40)->ShallowCopy(reader->GetOutput()->GetPointData());
	}

	printf("finished reading\n");

	lcsAxisAlignedFlowMap *flowMap = lcsAxisAlignedFlowMap::New();
	flowMap->SetDimensions(200, 200, 200);
	flowMap->SetXRange(-1.7, -0.87);
	flowMap->SetYRange(-0.728, 0.00849);
	flowMap->SetZRange(11, 12.3);
	flowMap->SetTimeStep(0.0001);
	flowMap->SetAdvectionTime(grid->GetTimePoint(grid->GetNumberOfSamples() - 1));
	flowMap->SetInputData(grid);

	printf("finished flowMap setting\n");

	vtkSmartPointer<vtkXMLRectilinearGridWriter> writer = vtkSmartPointer<vtkXMLRectilinearGridWriter>::New();
	writer->SetInputConnection(flowMap->GetOutputPort());
	writer->SetFileName("flowmap.vtk");

	printf("before write\n");

	writer->Write();

	grid->Delete();
	flowMap->Delete();

	return 0;
}
