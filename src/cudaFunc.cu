#include "cudaFunc.h"

//using namespace std; 
/*file for implementing the cuda version of rasterization functions*/
//
tri_t* sendTrianglesToDevice(tri_t* triList,int size)
{
   tri_t* tri_d;
   cudaMalloc(&tri_d,sizeof(tri_t)*size);
   cudaMemcpy(tri_d,triList,sizeof(tri_t)*size,cudaMemcpyHostToDevice);
   return tri_d;
}
//Function for retrieving the converted tri_t after the kernel has been run
//assumes that the tri_return points to a malloced pointer for the given size
tri_t* retrieveTrianglesFromDevice(tri_t* triList_d,int size)
{
   tri_t* tri_r;
   tri_r = (tri_t*)malloc(sizeof(tri_t)*size);
   cudaMemcpy(tri_r,triList_d,sizeof(tri_t)*size,cudaMemcpyDeviceToHost);
   cudaFree(triList_d);
   return tri_r;
}
point_t* retrievePointsFromDevice(point_t*pointList,point_t* point_d,int size)
{
   point_t* point_r;
   point_r = (point_t*)malloc(sizeof(point_t)*size);
   cudaMemcpy(point_r,point_d,sizeof(point_t)*size,cudaMemcpyDeviceToHost);
   return point_r;

}

//function for sending the points to the device, returns a pointer to the mem
point_t* sendPointsToDevice(point_t* pointList,int size)
{
   point_t* point_d;
   cudaMalloc(&point_d,sizeof(point_t)*size);
   cudaMemcpy(point_d,pointList,sizeof(point_t)*size,cudaMemcpyHostToDevice);
   return point_d;

}

//function to run the entire convert process
tri_t* cudaConvertCoords(tri_t* triList,int size, int h, int w)
{
   tri_t* tri_d;// pointer for input list on device
   tri_t* tri_r;// pointer for holding the result of calc
   dim3 dimBlock(h/50 +1, w/50 +1);
   dim3 dimGrid(50,3);
   tri_d= sendTrianglesToDevice(triList,size);
   //cudaCoordinateCalc<<<dimBlock,dimGrid>>>(thrust::raw_pointer_cast(tri_d),tri_d.size(),
  // thrust::raw_pointer_cast(tri_r),w,h);
   

   return tri_d;
   
}

/*
//kernel for finding the new points after conversion
__global__ void cudaCoordinateCalc(tri_t* triList, int listSize
   ,tri_t* tri_d,int w_in, int h_in,vec_t scale = (vec_t)1.0)
{
   int location;
   location =blockIdx.x*50 +threadIdx.x;
   //checki if this thread is within range
   if(location >= listSize)
   {
      return;
   }
   int tpX,tpY; 
   int h= h_in;
   int w= w_in;
  // Convert x.
   float tmpX = triList[location].p[threadIdx.x].x + dim; // Shift.
   tpX = (int)(tmpX * (float)(w - 1) / 2 * (1.f / dim)); // Scale.
   // Convert y.
   float tmpY = y + dim; // Shift.
   tpY = (int)(tmpY * (float)(h - 1) / 2 * (1.f / dim)); // Scale.

   vec_t dim = scale;
   // Convert x.
   vec_t tmpX = triList[location].pt[threadIdx.y].coords.v[0] + dim; // Shift.
   tpX = (int)(tmpX * (vec_t)(w - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
   // Convert y.
   vec_t tmpY = triList[location].pt[threadIdx.y].coords.v[1] + dim; // Shift.
   tpY = (int)(tmpY * (vec_t)(h - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.



   ((tri_t)tri_d[location]).((point_t)pt[threadIdx.y]).px = tpX;
   tri_d[location].pt[threadIdx.y].py = tpY;
   return;
}
*/
