### AxisAlignedFlowMap

A VTK-style filter for computing flowmap on an axis-aligned grid

### Build as shared library

In the root folder, run the following.

      $ cmake .
      $ make

It will generate libAxisAlignedFlowMap.so in the same folder.

### Example

The Example folder contains an example of using the compiled shared library libAxisAlignedFlowMap.so.

When compiling, put

      main.cpp
      lcsUnstructuredGridWithTimeVaryingPointData.h
      lcsAxisAlignedFlowMap.h
      CMakeListsForUser.txt (it should be renamed as CMakeLists.txt)
      libAxisAlignedFlowMap.so

in the same folder, then enter the commands below.

      $ cmake .
      $ make

It will generate an executable called FlowMap.

### License

Its free and open source under GNU/zlib license. Please see [License.txt](https://github.com/linyufly/AxisAlignedFlowMap/blob/master/license.txt) for terms.

### Author
Mingcheng Chen  
University of Illinois, Urbana-Champaign
