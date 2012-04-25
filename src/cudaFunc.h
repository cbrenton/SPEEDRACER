//Header file for the rasterizer cuda functions
#include <stdio.h>
#include "vector.h"
#include <vector>
#include "tri_t.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;


//function that manages the entire conversion process
__global__ void cudaCoordinateCalc(tri_t* triList,int listSize, tri_t* tri_d,int w_in, int h_in);

//kernel for converting the coordinates
thrust::host_vector<tri_t> cudaConvertCoords(vector<tri_t> triList,int h, int w);

//function that sends the vector to the device
tri_t* sendTrianglesToDevice(tri_t* triList,int size);

point_t* sendPointsToDevice(point_t* pointList,int size);

//function that retrieves the data vector and returns a direct pointer to it
tri_t* retrieveTrianglesFromDevice(tri_t* triList_d,int size);

