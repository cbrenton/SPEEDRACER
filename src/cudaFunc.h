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
#include "colorbuffer.h"
#include "zbuffer.h"
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

//function to send points to the constant mem on the graphics card, returns a pointer to it
point_t* sendPointToDeviceConst(point_t* pointList,int size);

vec3_t* retrieveColorFromDevice(vec3_t* color_d,int size);
//device function for the dot product of two vectors
__device__ vec_t dot_d(vec_t* a, vec_t* b);

//function to send the zbuffer to the device
vec_t* sendZBufferToDevice(zbuffer* zbuff,int size);

//device function to check if there is a hit on the current triangle
__device__ bool cudaHit(tri_t* tri, point_t *ptList, int x, int y, vec_t *t, vec_t *bary);
//function to sub for lack of floats in atomic min operation
__device__ float atomicMin_f(float* val,float z);
//kernel for rastirzing the traingles
__global__ void cudaRasterizeKernel(tri_t* tri_d,int tri_size,point_t* point_d,vec3_t* color_d,
   vec_t* zbuff_d,int height,int* lock);

//function wrapper for managing the rasterization process
void cudaRasterize(tri_t* tri_d,int tri_size,point_t* point_d,int ptSize,colorbuffer* color_d,
   zbuffer* zbuff_d);

//device function for finding the determinant
__device__ vec_t det_d(vec_t* data);

//function for setting up the lock on the zbuffer and color buffer for use with semaphores 
//int* setupBuffLock(int size)

//function for testing the transfer of triangles to the device
tri_t* testTriangles(tri_t* input,int size);

//wrapper function to blur the image with one pass
vec3_t* cudaBlur(colorbuffer* color,int h, int w);

//kernel for blur
__global__ void cudaBlur(vec3_t* color_d, int h, int w, bool isVert);

//function to prevent illegal address calls for range
__device__ float3 sample_d(vec3_t *cbuf,int h,int w, int x, int y);

// vector add function for the device
__device__ float3 vecAdd_d(float3 a, float3 b);

inline __host__ __device__ float3 operator*(float3 a, float s)
{
       return make_float3(a.x * s, a.y * s, a.z * s);
}

#endif
