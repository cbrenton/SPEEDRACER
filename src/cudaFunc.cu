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
tri_t* retrieveTrianglesFromDevice(tri_t* tri_d,int size)
{
   tri_t* tri_r;
   tri_r = (tri_t*)malloc(sizeof(tri_t)*size);
   cudaMemcpy(tri_r,tri_d,sizeof(tri_t)*size,cudaMemcpyDeviceToHost);
   cudaFree(tri_d);
   return tri_r;
}
point_t* retrievePointsFromDevice(point_t* point_d,int size)
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
point_t* cudaConvertCoords(point_t* pointList,int size, int h, int w,vec_t scale)
{
   point_t* point_d;// pointer for holding the result of calc
   dim3 dimBlock(h/50 +1, w/50 +1);
   dim3 dimGrid(50,3);
   point_d= sendPointsToDevice(pointList,size);
   cudaCoordinateCalc<<<dimBlock,dimGrid>>>(point_d, size, h, w,scale);
   return retrievePointsFromDevice(point_d,size);
   
}


//kernel for finding the new points after conversion
__global__ void cudaCoordinateCalc(point_t* point_d, int listSize,int w_in, int h_in,vec_t scale )
{
   int location;
   location =blockIdx.x*50 +threadIdx.x;
   //checki if this thread is within range
   if(location >= listSize)
   {
      return;
   }
   scale = (vec_t) 1.00;
   int tpX,tpY; 
   int h= h_in;
   int w= w_in;
   vec_t dim = scale;
   // Convert x.
   vec_t tmpX = point_d[location].coords.v[0] + dim; // Shift.
   tpX = (int)(tmpX * (vec_t)(w - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
   // Convert y.
   vec_t tmpY = point_d[location].coords.v[1] + dim; // Shift. 
   tpY = (int)(tmpY * (vec_t)(h - 1) / 2 * ((vec_t)1.0 / dim)); // Scale.
   point_d[location].pX = tpX;
   point_d[location].pY = tpY;
   return;
}

