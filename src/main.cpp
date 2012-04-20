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

using namespace std;

#define FLT_MIN 1.1754E-38F
#define FLT_MAX 1.1754E+38F

//for computing the center point and extent of the model
float cx, cy, cz;
float max_x, max_y, max_z, min_x, min_y, min_z;
float max_extent;

//other globals
int GW;
int GH;
int display_mode;
int view_mode;
vector<point_t *> pointList;
vector<tri_t *> triList;

void readLine(char* str);
void readStream(istream& is);

//open the file for reading
void ReadFile(char* filename) {
   printf("Reading coordinates from %s\n", filename);

   ifstream in_f(filename);
   if (!in_f)  {
      printf("Could not open file %s\n", filename);
   } else {
      readStream(in_f);
   }
}

//process the input stream from the file
void readStream(istream& is)
{
   char str[256];
   for (;is;) {
      is >> ws;
      is.get(str,sizeof(str));
      if (!is) break;
      is.ignore(9999,'\n');
      readLine(str);
   }
}

//process each line of input save vertices and faces appropriately
point_t * findPt(int ndx) {
   for (int i = 0; i < (int)pointList.size(); i++) {
      //if (isNum(pointList.at(i), ndx)) {
         return pointList.at(i);
      //}
   }
   fprintf(stderr, "Error: vertex not found.\n");
   exit(EXIT_FAILURE);
}

void readLine(char* str) {
   int vi;
   float x, y, z;
   float r, g, b;
   int mat;

   if (str[0]=='#') return;
   //read a vertex or face
   if (str[0]=='V' && !strncmp(str,"Vertex ",7)) {

      if (sscanf(str,"Vertex %d %g %g %g",&vi,&x,&y,&z) !=4)
      {
         printf("an error occurred in reading vertices\n");
#ifdef _DEBUG
         exit(EXIT_FAILURE);
#endif
      }

      //TODO allocate an object to store the vertex or face
      //store the vertex in a collection
      point_t *newPt = new point_t(vi, x, y, z);
      //*newPt = {vi, x, y, z};
      pointList.push_back(newPt);


      //This code is house keeping to display in center of the scene
      cx += x;
      cy += y;
      cz += z;
      if (x > max_x) max_x = x; if (x < min_x) min_x = x;
      if (y > max_y) max_y = y; if (y < min_y) min_y = y;
      if (z > max_z) max_z = z; if (z < min_z) min_z = z;
   }
   else if (str[0]=='F' && !strncmp(str,"Face ",5)) {
      //TODO allocate an object to store the vertex or face
      point_t *tmpPt1, *tmpPt2, *tmpPt3;
      tmpPt1 = tmpPt2 = tmpPt3 = NULL;
      char* s=str+4;
      int fi=-1;
      for (int t_i = 0;;t_i++) {
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
         if (fi<0) { fi=j; continue; }
         // Otherwise, process the digit we've grabbed in j as a vertex index.
         // The first number will be the face id; the following are vertex ids.
         if (t_i == 1){
            // Store the first vertex in your face object
            tmpPt1 = findPt(j);
         }
         else if (t_i == 2){
            // Store the second vertex in your face object
            tmpPt2 = findPt(j);
         }
         else if (t_i == 3){
            // Store the third vertex in your face object
            tmpPt3 = findPt(j);
            tri_t *newTri = new tri_t(tmpPt1, tmpPt2, tmpPt3);
            triList.push_back(newTri);
         }
         // If there is more data to process break out
         if (*s =='{') break;
      }
      // Possibly process colors if the mesh has colors
      if (*s && *s =='{'){
         char *s1 = s+1;
         cout << "trying to parse color " << !strncmp(s1,"rgb",3) << endl;
         // If we're reading off a color
         if (!strncmp(s1,"rgb=",4)) {
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

void printFirstThree() {
   printf("First 3 vertices:\n");
   pointList.at(0)->print();
   pointList.at(1)->print();
   pointList.at(2)->print();
}

int main(int argc, char** argv) {
   // Initialization
   max_x = max_y = max_z = FLT_MIN;
   min_x = min_y = min_z = FLT_MAX;
   cx = cy = cz = 0;
   max_extent = 1.0;
   // Make sure a file to read is specified
   if (argc > 1) {
      printf("Using file %s\n", argv[1]);
      // Read in the mesh file specified
      ReadFile(argv[1]);
      // Only for debugging
      printFirstThree();

      // Once the file is parsed find out the maximum extent to center and scale mesh
      max_extent = max_x - min_x;
      if (max_y - min_y > max_extent) max_extent = max_y - min_y;

      // Divide by the number of vertices you read in!!!

      cx = cx/(float)triList.size();
      cy = cy/(float)triList.size();
      cz = cz/(float)triList.size();
   } else {
      cout << "format is: meshparser filename" << endl;
   }
}
