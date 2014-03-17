### AxisAlignedFlowMap

A VTK-style filter for computing flowmap on an axis-aligned grid

### Build as shared library

In the root folder, run the following.

      $ cmake .
      $ make

It will generate libAxisAlignedFlowMap.so in the same folder.

### Example

The main.cpp together with CMakeListsForUser.txt is an example of using the compiled shared library libAxisAlignedFlowMap.so.

When compiling, put main.cpp, CMakeListsForUser.txt (it should be renamed as CMakeLists.txt) and libAxisAlignedFlowMap.so in the same folder, then enter the commands below.

      $ cmake .
      $ make

It will generate an executable of main.

### License

Its free and open source under GNU/zlib license. Please see [License.txt](https://github.com/linyufly/AxisAlignedFlowMap/blob/master/license.txt) for terms.

### Author
Mingcheng Chen  
University of Illinois, Urbana-Champaign
