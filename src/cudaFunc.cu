#include "cudaFunc.h"

//using namespace std; 
/*file for implementing the cuda version of rasterization functions*/
//function for sending the vector to the device
//__constant__ vector<tri_t>* tr_d;
thrust::device_vector<tri_t> sendTrianglesToDevice(vector<tri_t> triList)
{
   thrust::device_vector<tri_t> tri_d = triList;
   return tri_d;
}
//Function for retrieving the converted tri_t after the kernel has been run
thrust::host_vector<tri_t> retrieveCoordinatesFromDevice(thrust::device_vector<tri_t> tri_d)
{
   thrust::host_vector<tri_t> tri_h= tri_d;
   //tri_t * ptr = thrust::raw_pointer_cast(&tri_h[0]);

   return tri_h;
}
//function to run the entire convert process
thrust::host_vector<tri_t> cudaConvertCoords(vector<tri_t> triList,int h, int w)
{
   thrust::device_vector<tri_t> tri_d;// vector for input list on device
   thrust::device_vector<tri_t> tri_r;// vector for holding the result of calc
   dim3 dimBlock(h/50 +1, w/50 +1);
   dim3 dimGrid(50,3);
   tri_d= sendTrianglesToDevice(triList);
   cudaCoordinateCalc<<<dimBlock,dimGrid>>>(thrust::raw_pointer_cast(tri_d),tri_d.size(),
   thrust::raw_pointer_cast(tri_r),w,h);
  
   return retrieveCoordinatesFromDevice(tri_r);
   
}


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
/*   // Convert x.
   float tmpX = triList[location].p[threadIdx.x].x + dim; // Shift.
   tpX = (int)(tmpX * (float)(w - 1) / 2 * (1.f / dim)); // Scale.
   // Convert y.
   float tmpY = y + dim; // Shift.
   tpY = (int)(tmpY * (float)(h - 1) / 2 * (1.f / dim)); // Scale.
*/
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

