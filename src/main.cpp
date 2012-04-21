//basic program to read in a mesh file (of .m format from H. Hoppe)
//Hh code modified by ZJW for csc 471

//Fall 2010 - base code for lab on 'drawing' a 3d model - students must define the objects to store the mesh and the draw routines before prog 3

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>

#include "point_t.h"
#include "tri_t.h"
#include "image.h"

#define WINDOW_W 800
#define WINDOW_H 600

using namespace std;

// Other globals
int GW;
int GH;
int display_mode;
int view_mode;
vector<point_t *> pointList;
vector<tri_t *> triList;

void convertCoords();
void printCoords();
void rasterize();
point_t * findPt(int ndx);
void readFile(char* filename);
void readLine(char* str);
void readStream(istream& is);

int main(int argc, char** argv)
{
   // Make sure a file to read is specified
   if (argc > 1)
   {
      printf("Using file %s\n", argv[1]);

      // Get the mesh from the specified file.
      readFile(argv[1]);

      // Convert triangle coordinates from world to screen.
      convertCoords();

      // Go SPEEDRACER GO!
      rasterize();
   }
   else
   {
      cout << "format is: meshparser filename" << endl;
   }
}

void convertCoords()
{
   for (int triNdx = 0; triNdx < (int)triList.size(); triNdx++)
   {
      triList[triNdx]->w2p(WINDOW_W, WINDOW_H);
   }
}

void rasterize()
{
   // Initialize the image.
   Image *im = new Image(WINDOW_H, WINDOW_W, "out.png");

   float t = -1.f;
   // For each pixel in the image:
   for (int x = 0; x < WINDOW_W; x++)
   {
      for (int y = 0; y < WINDOW_H; y++)
      {
         for (int triNdx = 0; triNdx < (int)triList.size(); triNdx++)
         {
            // Check for intersection.
            if (triList[triNdx]->hit(x, y, &t) == 1)
            {
               // Write to file.
               float3 color(0.f, 1.f, 0.f);
               im->writePixel(x, y, &color);
            }
         }
      }
   }
   im->close();
   delete im;
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
void readFile(char* filename)
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
      // Store the new triangle in your face collection
   }
}
