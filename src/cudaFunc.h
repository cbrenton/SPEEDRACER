#ifndef _CUDAFUNC_H
#define _CUDAFUNC_H

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
__global__ void cudaCoordinateCalc(point_t* point_d, int listSize, int w_in, int h_in, vec_t scale );

//kernel for converting the coordinates
point_t* cudaConvertCoords(point_t* pointList, int size, int h, int w, vec_t scale);

//function that sends the vector to the device
tri_t* sendTrianglesToDevice(tri_t* triList, int size);

point_t* sendPointsToDevice(point_t* pointList, int size);

//function that retrieves the data vector and returns a direct pointer to it
tri_t* retrieveTrianglesFromDevice(tri_t* triList_d, int size);

//function to retrieve the point array, returns a malloced result
point_t* retrievePointsFromDevice(point_t* point_d, int size);

#endif
