#include "cudaFunc.h"

//using namespace std; 
/*file for implementing the cuda version of rasterization functions*/
//
tri_t* sendTrianglesToDevice(tri_t* triList,int size)
{
   tri_t* tri_d;
   cudaMalloc(&tri_d,sizeof(tri_t)*size);
   cudaMemcpy(tri_d,triList,sizeof(tri_t)*size,cudaMemcpyHostToDevice);
   //cudaMemcpyToSymbol(tri_d,triList,sizeof(tri_t)*size);
   //printf("TEST\n");
   //printf("%d %d %d \n",tri_d->pt[0],tri_d->pt[1],tri_d->pt[2]);
   return tri_d;
}
tri_t* testTriangles(tri_t* input,int size)
{
   tri_t* tri_d;
   tri_d = sendTrianglesToDevice(input,size);
   return retrieveTrianglesFromDevice(tri_d,size);
}
//function for writing the converted points onto the graphics card
point_t* sendPointToDeviceConst(point_t* pointList,int size)
{
   point_t* point_d;
   cudaMalloc(&point_d,sizeof(point_t)*size);
   cudaMemcpy(point_d,pointList,sizeof(point_t)*size,cudaMemcpyHostToDevice);
   //cudaMemcpyToSymbol(point_d,pointList,sizeof(point_t)*size);
   return point_d;
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
   cudaFree(point_d);
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

vec3_t* sendColorToDevice(colorbuffer* colorbuff,int size)
{
   vec3_t* color_d;
   cudaMalloc(&color_d,sizeof(vec3_t)*size);
   cudaMemcpy(color_d,colorbuff->data,sizeof(vec3_t)*size,cudaMemcpyHostToDevice);
   return color_d;

}
vec3_t* retrieveColorFromDevice(vec3_t* color_d,int size)
{
   vec3_t* color_r;
   color_r = (vec3_t*)malloc(sizeof(vec3_t)*size);
   cudaMemcpy(color_r,color_d,sizeof(vec3_t)*size,cudaMemcpyDeviceToHost);
   return color_r;

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
//function to send the zbuffer to the device
vec_t* sendZBufferToDevice(zbuffer* zbuff,int size)
{
   vec_t* zbuff_d;
   cudaMalloc(&zbuff_d,sizeof(vec_t)*size);
   cudaMemcpy(zbuff_d,zbuff->data,sizeof(vec_t)*size,cudaMemcpyHostToDevice);
   return zbuff_d;

}



//kernel for finding the new points after conversion 
//****confirmed works***
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

//function for setting up the lock on the zbuffer and color buffer for use with semaphores 
int* setupBuffLock(int size)
{
    int* lock;
   cudaMalloc(&lock,sizeof(int)*size);
   return lock;
}

//function for rasterization,called by main
//HAS NOT BEEN TESTED********
void cudaRasterize(tri_t* tri,int tri_size,point_t* point_d,colorbuffer* cbuff,
   zbuffer* zbuff)
{
   vec3_t* color_d;//device color buffer
   int*lock;
   tri_t* tri_d;
   tri_t* temp;
   temp = testTriangles(tri,tri_size);
   printf("passed %d\n",temp->pt0);
   //printf("sanity check\n");
   tri_d = sendTrianglesToDevice(tri,tri_size);
   int buffsize = cbuff->w * cbuff->h;//calculate the buffer size
   color_d = sendColorToDevice(cbuff,buffsize);//setup the color buffer on device
   vec_t* zbuff_d;
   zbuff_d = sendZBufferToDevice(zbuff,buffsize);
   dim3 dimBlock(tri_size/30 +1);
   dim3 dimGrid(30,1);
   lock= setupBuffLock(buffsize);
   printf("just before the kernel\n");
   cudaRasterizeKernel<<<dimBlock,dimGrid>>>(tri_d,tri_size,point_d,color_d,zbuff_d,cbuff->h,lock);
   cbuff->data = retrieveColorFromDevice(color_d,buffsize);
   printf("just after the kernel\n");
   return;

}
//function run in each thread to rasterize with the given data
//**HAS NOT BEEN TESTED****
__global__ void cudaRasterizeKernel(tri_t* tri_d,int tri_size,point_t* point_d,vec3_t* color_d,
   vec_t* zbuff_d,int height,int* lock)
{
   //check if this thread is within range of applicable triangles
   if(tri_size < blockIdx.x *30 +threadIdx.x)
   {
      return;
   }
   
   tri_t *tri = &tri_d[blockIdx.x*30+threadIdx.x];//register for the current triangle value
//printf("TEST VALUE\n");
  // for (int x = tri->extents[0]; x < tri->extents[1]; x++)
   for(int x=0;x<600;x++)
   {
    //  for (int y = tri->extents[2]; y < tri->extents[3]; y++)
      for(int y=0;y<600;y++)
      {
         
         vec_t z = zbuff_d[x * height + y];
         vec_t t = FLT_MAX;
         vec_t bary[3];
         //printf("just before hit");
         int tmpi = tri_d[blockIdx.x*30+threadIdx.x].pt[0];
   //      printf("pt[0]: %d\n", tmpi);
         if (cudaHit(&tri_d[blockIdx.x*30+threadIdx.x],point_d,x,y,&t,bary))
         {
            // Check the z-buffer to see if this should be written.
            if (t > z)
            {
/*
               // Calculate the normal.
               vec_t normal[3] = {
                  tri->normal[0],
                  tri->normal[1],
                  tri->normal[2]
               };
               // Calculate the color (N dot L).
               vec_t colorMag = dot_d(normal, light);
*/
               vec_t colorMag = 1.f;
               if (colorMag < 0)
               {
                  colorMag *= -1.f;
               }
               // Clamp the color to (0.0, 1.0).
               colorMag = max((vec_t)0.f, min(colorMag, (vec_t)1.f));
               // Write to color buffer. may need changes to not have triple pointer
              __syncthreads();
              if(atomicMin_f((float*)&zbuff_d[x * height + y],z) == z)//check if current min
              {
                while(atomicAdd((float*)&lock[x * height + y],1.f) == 0)//check if current pos is open
                { 
                   //write to the buffer
                   color_d[x * height + y].v[0] = bary[0];
                   color_d[x * height + y].v[1] = bary[1];
                   color_d[x * height + y].v[2] = bary[2];

                   atomicExch(&lock[x * height + y],0);
                   break;
                }
              }
               // Write to z-buffer.
               //*z = t;
            }
         }
      }
   }
}
//function to subs for missing atomic min on floats
__device__ float atomicMin_f(float* val,float z)
{
   float temp = atomicExch(val,z);
   if(z<temp)//case where z was the min
   {
      return z;
   }
   else
   {
      atomicExch(val,temp);
      return temp;
   }
}

__device__ void printMat(vec_t *m)
{
printf("{%f %f %f\n%f %f %f\n%f %f %f}\n", m[0], m[1], m[2],
   m[3], m[4], m[5],
   m[6], m[7], m[8]);
}

/*__device__ bool cudaHit(int x, int y, vec_t *t, vec_t *bary,tri_t* tri_d,point_t* point_d,int index*/
__device__ bool cudaHit(tri_t* tri, point_t *ptList, int x, int y, vec_t *t, vec_t *bary)
{
   //printf("hit started\n");
   //if (x < tri->extents[0] || x > tri->extents[1] ||
    //     y < tri->extents[2] || y > tri->extents[3])
     // return false;
   //printf("passed the test\n");
   bool hit = true;

   vec_t bBeta, bGamma, bT;

   vec_t pix[3] = {(vec_t)x, (vec_t)y, 0.f};
   vec_t screenPt[3][3];
//printf("pt[0]: %d\n", tri->pt[0]);
   for (int i = 0; i < 3; i++)
   {
      screenPt[i][0] = (vec_t)ptList[tri->pt[i]].pX;
      screenPt[i][1] = (vec_t)ptList[tri->pt[i]].pY;
      screenPt[i][2] = 0.f;
   }

   vec_t A[9] = {screenPt[0][0], screenPt[1][0], screenPt[2][0],
      screenPt[0][1], screenPt[1][1], screenPt[2][1],
      1.f, 1.f, 1.f};

   vec_t detA = det_d(A);
//printMat(A);
//printf("oh yeeeeaaaahhh\n");
//printf("%d\n", detA);
   if (detA == 0)
   {
      return false;
   }

   vec_t baryT[9] = {pix[0], screenPt[1][0], screenPt[2][0],
      pix[1], screenPt[1][1], screenPt[2][1],
      1.f, 1.f, 1.f};

   bT = det_d(baryT) / detA;

   if (bT < 0)
   {
      hit = false;
   }
   else
   {
      vec_t baryGamma[9] = {screenPt[0][0], pix[0], screenPt[2][0],
         screenPt[0][1], pix[1], screenPt[2][1],
         1.f, 1.f, 1.f};

      bGamma = det_d(baryGamma) / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         hit = false;
      }
      else
      {
         vec_t baryBeta[9] = {screenPt[0][0], screenPt[1][0], pix[0],
            screenPt[0][1], screenPt[1][1], pix[1],
            1.f, 1.f, 1.f};

         bBeta = det_d(baryBeta) / detA;

         if (bBeta < 0 || bBeta > 1 - bGamma)
         {
            hit = false;
         }
      }
   }

   if (hit)
   {
      *t = bT * ptList[tri->pt[0]].coords.v[2] + bBeta * ptList[tri->pt[1]].coords.v[2] + bGamma *
         ptList[tri->pt[2]].coords.v[2];
      if (bary)
      {
         bary[0] = bT;
         bary[1] = bBeta;
         bary[2] = bGamma;
      }
   }
   return hit;
}
//function that increments across all the pixels in range and returns if it is hit
/*__device__ bool cudaHit(int x, int y, vec_t *t, vec_t *bary,tri_t* tri_d,point_t* point_d,
   int index) 
{
   bool hit=false;
   int i,j;
   //loop over the range of x pixels
   for(i=tri_d->extents[0]; i<tri_d->extents[1];i++)
   {
      //loop over the range of y pixels
     for(j=tri_d->extents[2];j<tri_d->extents[3];j++)
     {
        //logic for hit
     }
 
   }

   return hit;
}
*/
__device__ vec_t dot_d(vec_t* a, vec_t* b)
{
   return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ vec_t det_d(vec_t* data)
{
   return data[0 * 3 + 0] * data[1 * 3 + 1] * data[2 * 3 + 2] + data[0 * 3 + 1] * data[1 * 3 + 2] *
      data[2 * 3 + 0] + data[0 * 3 + 2] * data[1 * 3 + 0] * data[2 * 3 + 1] - data[0 * 3 + 2] *
      data[1 * 3 + 1] * data[2 * 3 + 0] - data[0 * 3 + 0] * data[1 * 3 + 2] * data[2 * 3 + 1] -
      data[0 * 3 + 1] * data[1 * 3 + 0] * data[2 * 3 + 2];
}
