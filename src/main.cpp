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
#include "vec3.h"
#include "zbuffer.h"

// Default values.
#define DEF_W 800
#define DEF_H 600
#define DEF_OUTFILE "out.png"
#define DEF_SCALE
#define DEF_PROGRESS true

using namespace std;

int width = DEF_W;
int height = DEF_H;
vector<point_t *> pointList;
vector<tri_t *> triList;
vec3 light (0, 0, 1);
vec_t scale = 1.f;
bool showProgress = DEF_PROGRESS;
struct timeval startTime;

void convertCoords();
void printCoords();
void rasterize(string outName);
void rasterizePixel(vector<tri_t *> *tris, int x, int y, vec_t *z,
      vec3 *color, string outName);
point_t * findPt(int ndx);
void readFile(const char* filename);
void readLine(char* str);
void readStream(istream& is);

int main(int argc, char** argv)
{
   // Initialize.
   string inFile;
   string outFile;
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
         outFile = optarg;
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
   
   // Set up timekeeping.
   gettimeofday(&startTime, NULL);

   // Make sure a file to read is specified
   if (inputSpecified)
   {
      if (!outputSpecified)
      {
         outFile = DEF_OUTFILE;
      }
      // Get the mesh from the specified file.
      readFile(inFile.c_str());

      // Convert triangle coordinates from world to screen.
      convertCoords();

      // Go SPEEDRACER go!
      rasterize(outFile);
   }
   else
   {
      cout << "format is: meshparser filename" << endl;
   }
}

void convertCoords()
{
   for (int pointNdx = 0; pointNdx < (int)pointList.size(); pointNdx++)
   {
      pointList[pointNdx]->w2p(width, height, scale);
   }
}

void rasterize(string outName)
{
   // Initialize the image.
   Image *im = new Image(height, width, outName);
   zbuffer *zbuf = new zbuffer(width, height);
   colorbuffer *cbuf = new colorbuffer(width, height);

   // For each pixel in the image:
   for (int x = 0; x < width; x++)
   {
      for (int y = 0; y < height; y++)
      {
         vec_t *z = zbuf->data[x][y];
         vec3 *color = &cbuf->data[x][y];
         // Rasterize the current pixel.
         rasterizePixel(&triList, x, y, z, color, outName);
#ifndef _CUDA
         // Print out progress bar.
         if (showProgress)
         {
            // Set the frequency of ticks to update every .01%, if possible.
            int tick = max(width * height / 10000, 100);
            printProgress(startTime, x * height + y,
                  width * height, tick);
         }
#endif
      }
   }
   // Write the color buffer to the image file.
   im->write(cbuf);
   // Close image and clean up.
   im->close();
   delete im;
   delete cbuf;
   delete zbuf;
}

void rasterizePixel(vector<tri_t *> *tris, int x, int y, vec_t *z,
      vec3 *color, string outName)
{
   for (int triNdx = 0; triNdx < (int)tris->size(); triNdx++)
   {
      tri_t *tri = tris->at(triNdx);
      // Check for intersection.
      vec_t t = -1.f;
      if (tri->hit(x, y, &t))
      {
         // Check the z-buffer to see if this should be written.
         if (t < *z)
         {
            // Calculate the normal.
            vec3 ab = tri->pt[0]->toF3World() - tri->pt[1]->toF3World();
            vec3 ac = tri->pt[0]->toF3World() - tri->pt[2]->toF3World();
            vec3 normal = ab.cross(ac);
            normal.normalize();
            // Calculate the color (N dot L).
            vec_t colorMag = normal.dot(light);
            if (colorMag < 0)
            {
               colorMag *= -1.f;
            }
            // Clamp the color to (0.0, 1.0).
            colorMag = max((vec_t)0.f, min(colorMag, (vec_t)1.f));
            // Write to color buffer.
            color->v[0] = color->v[1] = color->v[2] = colorMag;
            // Write to z-buffer.
            *z = t;
         }
      }
   }
}

// Process each line of input save vertices and faces appropriately
point_t * findPt(int ndx)
{
   for (int i = 0; i < (int)pointList.size(); i++)
   {
      if (pointList.at(i)->isNum(ndx))
      {
         return pointList.at(i);
      }
   }
   fprintf(stderr, "Error: vertex not found.\n");
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
   vec_t x, y, z;
   vec_t r, g, b;
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
      point_t *newPt = new point_t(vi, x, y, z);
      pointList.push_back(newPt);
   }
   else if (str[0]=='F' && !strncmp(str,"Face ",5))
   {
      point_t *tmpPt1, *tmpPt2, *tmpPt3;
      tmpPt1 = tmpPt2 = tmpPt3 = NULL;
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
            // Store the third vertex in your face object
            tmpPt3 = findPt(j);
            tri_t *newTri = new tri_t(tmpPt1, tmpPt2, tmpPt3);
            // Store the new triangle in your face collection
            triList.push_back(newTri);
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
