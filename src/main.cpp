/**
 * SPEEDRACER: A software rasterizer created for Zoe Wood's CSC 572 class
 *    in conjunction with Chris Lupo's CPE 458 class.
 * @authors Chris Brenton, David Burke
 * @date Spring 2012
 *
 * GO SPEEDRACER GO!
 */
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>

#include "colorbuffer.h"
#include "image.h"
#include "point_t.h"
#include "progress.h"
#include "tri_t.h"
#include "vector.h"
#include "zbuffer.h"

#ifdef USE_CUDA
#include "cudaFunc.h"
#endif

// Default values.
#define DEF_W 1600
#define DEF_H 1600
#define DEF_OUTFILE "out.png"
#define DEF_SCALE
#define DEF_PROGRESS true

#define NUM_BUNNIES 1
#define NUM_BLURS 100

using namespace std;

int width = DEF_W;
int height = DEF_H;
vector<point_t> pointList;
vector<tri_t> triList;
int pointSize;
int triSize;
tri_t *triArray;
point_t *pointArray;
vec_t light[3] = {0.f, 0.f, 1.f};
vec_t scale = 1.f;
bool showProgress = DEF_PROGRESS;
string outName;

#ifdef USE_CUDA
void gpuConvertCoords();
#endif
void convertCoords();
void pointsToArray();
void trisToArray();
void makeBoundingBoxes();
void makeNormals();
void printCoords();
void rasterize();
void blurIt(colorbuffer *cbuf);
//colorbuffer * blur(colorbuffer *cbuf, bool isVert);
vec3_t * blur(colorbuffer *cbuf, bool isVert);
vec3_t sample(colorbuffer *cbuf, int x, int y);
void rasterizeTri(tri_t *tris, int triSize, colorbuffer *cbuf, zbuffer *zbuf);
bool cpuHit(tri_t tri, point_t *ptList, int ptSize, int x, int y, vec_t *t, vec_t *bary);
vec_t dot_h(vec_t *a, vec_t *b);
vec_t det_h(vec_t *data);
int findPt(int ndx);
void readFile(const char* filename);
void readLine(char* str);
void readStream(istream& is);

int main(int argc, char** argv)
{
   // Initialize.
   string inFile;
   bool inputSpecified = false;
   bool outputSpecified = false;
   int c;

   // Get command line arguments.
   while ((c = getopt(argc, argv, "h:i:o:ps:w:")) != -1)
   {
      switch (c)
      {
      case 'i':
         inFile = optarg;
         inputSpecified = true;
         break;
      case 'o':
         outName = optarg;
         outputSpecified = true;
         break;
      case 'w':
         width = atoi(optarg);
         break;
      case 'h':
         height = atoi(optarg);
         break;
      case 's':
         scale = (vec_t)atof(optarg);
         break;
      case 'p':
         showProgress = !showProgress;
         break;
      case '?':
         if (optopt == 'i' || optopt == 'o')
         {
            fprintf(stderr, "Option -%c requires a filename argument.\n",
                  optopt);
         }
         break;
      default:
         fprintf(stderr, "?\n");
         break;
      }
   }

#ifdef USE_CUDA
   printf("Using CUDA.\n");
#else
   printf("Not using CUDA.\n");
#endif

   // Set up timekeeping.
   initProgress();

   // Make sure a file to read is specified
   if (inputSpecified)
   {
      if (!outputSpecified)
      {
         outName = DEF_OUTFILE;
      }
      // Get the mesh from the specified file.
      readFile(inFile.c_str());

      // Convert point vector to an array.
      pointsToArray();

      // Convert triangle coordinates from world to screen.
      //#ifdef USE_CUDA
      //gpuConvertCoords();
      //#else
      convertCoords();
      //#endif

      // Generate triangle bounding boxes.
      makeBoundingBoxes();

      // Generate triangle normals.
      makeNormals();

      // Convert triangle vector to an array.
      trisToArray();

      // Go SPEEDRACER go!
      rasterize();
   }
   else
   {
      cout << "format is: meshparser filename" << endl;
   }
}

#ifdef USE_CUDA
void gpuConvertCoords()
{
   pointSize = (int)pointList.size();
   pointArray = cudaConvertCoords(pointArray, pointSize,
         height, width, scale);
}
#endif

void convertCoords()
{
   pointSize = (int)pointList.size();
   for (int pointNdx = 0; pointNdx < pointSize; pointNdx++)
   {
      pointArray[pointNdx].w2p(width, height, scale);
   }
}

void pointsToArray()
{
   pointSize = (int)pointList.size();
   pointArray = new point_t [pointSize];
   for (int point = 0; point < pointSize; point++)
   {
      pointArray[point] = pointList[point];
   }
   for (int triNdx = 0; triNdx < (int)triList.size(); triNdx++)
   {
      triList[triNdx].ptList = pointArray;
      triList[triNdx].numPts = pointSize;;
   }
}

void trisToArray()
{
   triSize = (int)triList.size();
   triArray = new tri_t[triSize];
   for (int tri = 0; tri < triSize; tri++)
   {
      triArray[tri] = triList[tri];
   }
}

void makeBoundingBoxes()
{
   for (int triNdx = 0; triNdx < (int)triList.size(); triNdx++)
   {
      triList[triNdx].genExtents(width, height);
      //triList[triNdx].debug();
   }
}

void makeNormals()
{
   for (int triNdx = 0; triNdx < (int)triList.size(); triNdx++)
   {
      triList[triNdx].genNormal();
   }
}

void rasterize()
{
   // Initialize the image.
   Image *im = new Image(width, height, outName);
   zbuffer *zbuf = new zbuffer(width, height);
   colorbuffer *cbuf = new colorbuffer(width, height);

//#ifdef USE_CUDA
   //cudaRasterize(triArray, triSize, pointArray,pointList.size(), cbuf, zbuf);
//#else
   rasterizeTri(triArray, triSize, cbuf, zbuf);
//#endif

   // Blur it!
   blurIt(cbuf);
   
   // Rasterize again, for "bunny, thinking of bunny" image.
   //zbuf->clear();

   //rasterizeTri(triArray, triSize, cbuf, zbuf);

   // Write the color buffer to the image file.
   im->write(cbuf);
   // Close image and clean up.
   delete im;
   delete cbuf;
   delete zbuf;
}

void rasterizeTri(tri_t *tris, int triSize, colorbuffer *cbuf, zbuffer *zbuf)
{
   printf("Rasterizing.\n");
   int h = cbuf->h;
   for (int triNdx = 0; triNdx < triSize; triNdx++)
   {
      tri_t tri = tris[triNdx];
      for (int x = tri.extents[0]; x < tri.extents[1]; x++)
      {
         for (int y = tri.extents[2]; y < tri.extents[3]; y++)
         {
            vec_t* z = &zbuf->data[x * h + y];
            vec_t t = FLT_MAX;
            vec_t bary[3] = {1.f, 0.f, 0.f};
            if (cpuHit(tri, pointArray, pointSize, x, y, &t, bary))
            {
               // Check the z-buffer to see if this should be written.
               if (t > *z)
               {
                  // Calculate the normal.
                  vec_t *normal = tri.normal;
                  // Calculate the color (N dot L).
                  vec_t colorMag = dot_h(normal, light);
                  if (colorMag < 0)
                  {
                     colorMag *= -1.f;
                  }
                  // Clamp the color to (0.0, 1.0).
                  colorMag = max((vec_t)0.f, min(colorMag, (vec_t)1.f));
                  // Write to color buffer.
                  cbuf->data[x * h + y][0] = colorMag;
                  cbuf->data[x * h + y][1] = colorMag;
                  cbuf->data[x * h + y][2] = colorMag;
                  // Write to z-buffer.
                  *z = t;
               }
            }
         }
      }
   }
}

bool cpuHit(tri_t tri, point_t *ptList, int ptSize, int x, int y, vec_t *t, vec_t *bary)
{
   if (x < tri.extents[0] || x > tri.extents[1] ||
         y < tri.extents[2] || y > tri.extents[3])
      return false;

   bool hit = true;

   vec_t bBeta, bGamma, bT;

   vec_t pix[3] = {(vec_t)x, (vec_t)y, 0.f};
   vec_t screenPt[3][3];
   for (int i = 0; i < 3; i++)
   {
      screenPt[i][0] = (vec_t)ptList[tri.pt[i]].pX;
      screenPt[i][1] = (vec_t)ptList[tri.pt[i]].pY;
      screenPt[i][2] = 0.f;
   }
   /* This is for our broke-ass CUDA version.
      screenPt[0][0] = (vec_t)ptList[tri.pt0].pX;
      screenPt[0][1] = (vec_t)ptList[tri.pt0].pY;
      screenPt[0][2] = 0.f;


      screenPt[1][0] = (vec_t)ptList[tri.pt1].pX;
      screenPt[1][1] = (vec_t)ptList[tri.pt1].pY;
      screenPt[1][2] = 0.f;


      screenPt[2][0] = (vec_t)ptList[tri.pt2].pX;
      screenPt[2][1] = (vec_t)ptList[tri.pt2].pY;
      screenPt[2][2] = 0.f;
      */

   vec_t A[9] = {screenPt[0][0], screenPt[1][0], screenPt[2][0],
      screenPt[0][1], screenPt[1][1], screenPt[2][1],
      1.f, 1.f, 1.f};

   vec_t detA = det_h(A);
   if (detA == 0)
   {
      return false;
   }

   vec_t baryT[9] = {pix[0], screenPt[1][0], screenPt[2][0],
      pix[1], screenPt[1][1], screenPt[2][1],
      1.f, 1.f, 1.f};

   bT = det_h(baryT) / detA;

   if (bT < 0)
   {
      hit = false;
   }
   else
   {
      vec_t baryGamma[9] = {screenPt[0][0], pix[0], screenPt[2][0],
         screenPt[0][1], pix[1], screenPt[2][1],
         1.f, 1.f, 1.f};

      bGamma = det_h(baryGamma) / detA;

      if (bGamma < 0 || bGamma > 1)
      {
         hit = false;
      }
      else
      {
         vec_t baryBeta[9] = {screenPt[0][0], screenPt[1][0], pix[0],
            screenPt[0][1], screenPt[1][1], pix[1],
            1.f, 1.f, 1.f};

         bBeta = det_h(baryBeta) / detA;

         if (bBeta < 0 || bBeta > 1 - bGamma)
         {
            hit = false;
         }
      }
   }

   if (hit)
   {
      //*t = bT * ptList[tri.pt[0]].coords.v[2] + bBeta * ptList[tri.pt[1]].coords.v[2] + bGamma *
      //ptList[tri.pt[2]].coords.v[2];
      *t = bT * ptList[tri.pt[0]].coords.v[2] + bBeta * ptList[tri.pt[1]].coords.v[2] + bGamma *
         ptList[tri.pt[2]].coords.v[2];
      if (bary)
      {
         bary[0] = bT;
         bary[1] = bBeta;
         bary[2] = bGamma;
      }
   }
   return hit;
}

void blurIt(colorbuffer *cbuf)
{
   printf("Blurring.\n");
   for (int i = 0; i < NUM_BLURS; i++)
   {
#ifdef USE_CUDA
      printf("CUDA BLUR\n");
      cbuf->data = cudaBlur(cbuf, cbuf->h, cbuf->w);
#else
      // Blur horizontally.
      cbuf->data = blur(cbuf, false);
      // Blur vertically.
      cbuf->data = blur(cbuf, true);
#endif
   }
}

vec3_t *blur(colorbuffer *cbuf, bool isVert)
{
   vec_t blurWeights[9] = {
      1.f / 256.f,
      8.f / 256.f,
      28.f / 256.f,
      56.f / 256.f,
      70.f / 256.f,
      56.f / 256.f,
      28.f / 256.f,
      8.f / 256.f,
      1.f / 256.f
   };
   int size = 9;

   vec3_t *ret = new vec3_t[cbuf->w * cbuf->h];
   //colorbuffer *ret = new colorbuffer(cbuf->w, cbuf->h);

   for (int x = 0; x < cbuf->w; x++)
   {
      for (int y = 0; y < cbuf->h; y++)
      {
         vec3_t result;
         vec3_t *samples = new vec3_t[size];
         int start = - size / 2;
         for (int n = 0; n < size; n++)
         {
            int offset = start + n;
            if (isVert)
            {
               samples[n] = sample(cbuf, x, y + offset);
            }
            else
            {
               samples[n] = sample(cbuf, x + offset, y);
            }
            samples[n] *= blurWeights[n];
            result += samples[n];
            result.clamp(0, 1);
         }
         ret[x * cbuf->h + y] = result;
         delete [] samples;
      }
   }
   return ret;
}

vec3_t sample(colorbuffer *cbuf, int x, int y)
{
   int newX = x;
   int newY = y;
   if (x < 0)
      newX = 0;
   if (x > cbuf->w)
      newX = cbuf->w;
   if (y < 0)
      newY = 0;
   if (y > cbuf->h)
      newY = cbuf->h;
   return cbuf->data[newX * cbuf->h + newY];
}

vec_t dot_h(vec_t *a, vec_t *b)
{
   return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

vec_t det_h(vec_t *data)
{
   return data[0 * 3 + 0] * data[1 * 3 + 1] * data[2 * 3 + 2] + data[0 * 3 + 1] * data[1 * 3 + 2] *
      data[2 * 3 + 0] + data[0 * 3 + 2] * data[1 * 3 + 0] * data[2 * 3 + 1] - data[0 * 3 + 2] *
      data[1 * 3 + 1] * data[2 * 3 + 0] - data[0 * 3 + 0] * data[1 * 3 + 2] * data[2 * 3 + 1] -
      data[0 * 3 + 1] * data[1 * 3 + 0] * data[2 * 3 + 2];
}

// Process each line of input save vertices and faces appropriately
int findPt(int ndx)
{
   for (int i = 0; i < (int)pointList.size(); i++)
   {
      if (pointList.at(i).isNum(ndx))
      {
         return i;
      }
   }
   fprintf(stderr, "Error: vertex %d not found.\n", ndx);
   exit(EXIT_FAILURE);
}

// Open the file for reading
void readFile(const char* filename)
{
   printf("Reading coordinates from %s\n", filename);

   ifstream in_f(filename);
   if (!in_f)
   {
      printf("Could not open file %s\n", filename);
   }
   else
   {
      readStream(in_f);
   }
}

// Process the input stream from the file
void readStream(istream& is)
{
   char str[256];
   for (;is;)
   {
      is >> ws;
      is.get(str,sizeof(str));
      if (!is) break;
      is.ignore(9999,'\n');
      readLine(str);
   }
}

void readLine(char* str)
{
   int vi;
   float x, y, z;
   float r, g, b;
   int mat;

   if (str[0]=='#') return;
   // Read a vertex or face
   if (str[0]=='V' && !strncmp(str,"Vertex ",7))
   {

      if (sscanf(str,"Vertex %d %g %g %g",&vi,&x,&y,&z) !=4)
      {
         printf("an error occurred in reading vertices\n");
#ifdef _DEBUG
         exit(EXIT_FAILURE);
#endif
      }

      // Store the vertex in a collection
      point_t newPt(vi, x, y, z);
      pointList.push_back(newPt);
   }
   else if (str[0]=='F' && !strncmp(str,"Face ",5))
   {
      int tmpPt1 = -1;
      int tmpPt2 = -1;
      int tmpPt3 = -1;
      char* s=str+4;
      int fi=-1;
      for (int t_i = 0;;t_i++)
      {
         while (*s && isspace(*s)) s++;
         // If we reach the end of the line break out of the loop
         if (!*s) break;
         // Save the position of the current character
         char* beg=s;
         // Advance to next space
         while (*s && isdigit(*s)) s++;
         // Convert the character to an integer
         int j=atoi(beg);
         // The first number we encounter will be the face index, don't store it
         if (fi<0)
         { fi=j; continue; }
         // Otherwise, process the digit we've grabbed in j as a vertex index.
         // The first number will be the face id; the following are vertex ids.
         if (t_i == 1)
         {
            // Store the first vertex in your face object
            tmpPt1 = findPt(j);
         }
         else if (t_i == 2)
         {
            // Store the second vertex in your face object
            tmpPt2 = findPt(j);
         }
         else if (t_i == 3)
         {
            // Do this 25 times.
            for (int i = 0; i < NUM_BUNNIES; i++)
            {
               // Store the third vertex in your face object
               tmpPt3 = findPt(j);
               tri_t newTri(tmpPt1, tmpPt2, tmpPt3, pointArray,
                     pointSize);
               // Store the new triangle in your face collection
               triList.push_back(newTri);
            }
         }
         // If there is more data to process break out
         if (*s =='{') break;
      }
      // Possibly process colors if the mesh has colors
      if (*s && *s =='{')
      {
         char *s1 = s+1;
         cout << "trying to parse color " << !strncmp(s1,"rgb",3) << endl;
         // If we're reading off a color
         if (!strncmp(s1,"rgb=",4))
         {
            // Grab the values of the string
            if (sscanf(s1,"rgb=(%g %g %g) matid=%d",&r,&g,&b,&mat)!=4)
            {
               printf("error during reading rgb values\n");
#ifdef _DEBUG
               exit(EXIT_FAILURE);
#endif
            }
         }
      }
   }
}
