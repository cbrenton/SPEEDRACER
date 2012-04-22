//Header file for the rasterizer cuda functions
#include <stdio.h>
#include "vec3.h"
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
thrust::device_vector<tri_t> sendTrianglesToDevice(vector<tri_t> triList);

//function that retrieves the data vector and returns a direct pointer to it
thrust::host_vector<tri_t> retrieveCoordinatesFromDevice(thrust::device_vector<tri_t> tri_d);

