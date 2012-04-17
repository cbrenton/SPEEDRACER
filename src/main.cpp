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
#include <map>
#include <GL/glut.h>
#include <GL/gl.h>

#ifndef __POINT_H
#define __POINT_H
#include "point.h"
#endif

#ifndef __FACE_H
#define __FACE_H
#include "face.h"
#endif

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
vector<Point> pointList;
vector<Face> faceList;

//TODO define objects to store 1) vertices 2) faces - be sure you understand the file format
//Point tmpPt;
//Face tmpFace;

//forward declarations of functions
void readLine(char* str);
void readStream(istream& is);
//void display();

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
Point findPt(int ndx) {
  for (int i = 0; i < (int)pointList.size(); i++) {
    if (pointList.at(i).isNum(ndx)) {
      return pointList.at(i);
    }
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
    //tmpPt.update(vi, x, y, z);
    //pointList.push_back(tmpPt);
    pointList.push_back(Point(vi, x, y, z));


    //This code is house keeping to display in center of the scene
    cx += x;
    cy += y;
    cz += z;
    if (x > max_x) max_x = x; if (x < min_x) min_x = x;
    if (y > max_y) max_y = y; if (y < min_y) min_y = y;
    if (z > max_z) max_z = z; if (z < min_z) min_z = z;
  } else if (str[0]=='F' && !strncmp(str,"Face ",5)) {
    //TODO allocate an object to store the vertex or face
    //Face tmpFace = new Face();
    Point tmpPt1 = Point();
    Point tmpPt2 = Point();
    Point tmpPt3 = Point();
    char* s=str+4;
    int fi=-1;
    for (int t_i = 0;;t_i++) {
      while (*s && isspace(*s)) s++;
      //if we reach the end of the line break out of the loop
      if (!*s) break;
      //save the position of the current character
      char* beg=s;
      //advance to next space
      while (*s && isdigit(*s)) s++;
      //covert the character to an integer
      int j=atoi(beg);
      //the first number we encounter will be the face index, don't store it
      if (fi<0) { fi=j; continue; }
      //otherwise process the digit we've grabbed in j as a vertex index
      //the first number will be the face id the following are vertex ids
      if (t_i == 1){
        //TODO store the first vertex in your face object
        //tmpFace.addV1(pointList.at(fi);
        tmpPt1 = findPt(j);
      }else if (t_i == 2){
        //TODO store the second vertex in your face object
        //tmpFace.addV2(pointList.at(fi);
        tmpPt2 = findPt(j);
      }else if (t_i == 3){
        //TODO store the third vertex in your face object
        //tmpFace.addV3(pointList.at(fi);
        tmpPt3 = findPt(j);
        faceList.push_back(Face(tmpPt1, tmpPt2, tmpPt3));
      }
      //faceList.push_back(tmpFace);
      //if there is more data to process break out
      if (*s =='{') break;
    }
    //possibly process colors if the mesh has colors
    if (*s && *s =='{'){
      char *s1 = s+1;
      cout << "trying to parse color " << !strncmp(s1,"rgb",3) << endl;
      //if we're reading off a color
      if (!strncmp(s1,"rgb=",4)) {
        //grab the values of the string
        if (sscanf(s1,"rgb=(%g %g %g) matid=%d",&r,&g,&b,&mat)!=4)
        {
          printf("error during reading rgb values\n");
#ifdef _DEBUG
          exit(EXIT_FAILURE);
#endif
        }
      }
    }
    //store the triangle read in to your face collection

  }
}

//TODO implement a simple routine that prints the first three vertices and faces to test that you successfully stored the data your data structures....
void printFirstThree() {
  printf("first 3 vertices:\n");
  pointList.at(0).print();
  pointList.at(1).print();
  pointList.at(2).print();
}

//TODO imlement a drawing routine to draw triangles as wireframe
void drawTria(Face f) {
  glBegin(GL_LINE_LOOP);
  //glBegin(GL_TRIANGLES);
  glColor3f(0.0, 0.0, 0.5);
  f.draw();
  glEnd();
}

void drawSphere() {
  glColor3f(1.0, 0.0, 0.0);
  glutWireSphere(0.35, 10, 10);
}


void drawObjects() {

  //transforms for the mesh
  glPushMatrix();
  //leave these transformations in as they center and scale each mesh correctly
  //scale object to window
  glScalef(1.0/(float)max_extent, 1.0/(float)max_extent, 1.0/(float)max_extent);
  //translate the object to the origin
  glTranslatef(-(cx), -(cy), -(cz));
  //TODO finish code to draw the wireframe mesh
  for(unsigned int j = 0; j < faceList.size(); j++) {
    drawTria(faceList.at(j));
  }
  glPopMatrix();

  //transforms for the sphere
  glPushMatrix();
  //draw the glut sphere behind the mesh
  glTranslatef(1.25, 0.0, -2.0);
  drawSphere();
  glPopMatrix();
}

void display() {


  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);

  glPushMatrix();
  //set up the camera
  gluLookAt(0, 0, 3.0, 0, 0, 0, 0, 1, 0);

  drawObjects();

  glPopMatrix();

  glutSwapBuffers();

}

void reshape(int w, int h) {
  GW = w;
  GH = h;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho( -2.0*(float)w/h, 2.0*(float)w/h, -2.0, 2.0, 1.0, 15.0);
  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);

  glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y ){
  switch( key ) {
  case 'q' : case 'Q' :
    exit( EXIT_SUCCESS );
    break;
  }
}

int main( int argc, char** argv ) {

  //set up my window
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(300, 300);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("IT'S DRAGON TIME");
  glClearColor(1.0, 1.0, 1.0, 1.0);

  //register glut callback functions
  glutDisplayFunc( display );
  glutReshapeFunc( reshape );
  glutKeyboardFunc( keyboard );
  //enable z-buffer
  glEnable(GL_DEPTH_TEST);

  //initialization
  max_x = max_y = max_z = FLT_MIN;
  min_x = min_y = min_z = FLT_MAX;
  cx =cy = cz = 0;
  max_extent = 1.0;
  //make sure a file to read is specified
  if (argc > 1) {
    cout << "file " << argv[1] << endl;
    //read-in the mesh file specified
    ReadFile(argv[1]);
    //only for debugging
    printFirstThree();

    //once the file is parsed find out the maximum extent to center and scale mesh
    max_extent = max_x - min_x;
    if (max_y - min_y > max_extent) max_extent = max_y - min_y;
    //cout << "max_extent " << max_extent << " max_x " << max_x << " min_x " << min_x << endl;
    //cout << "max_y " << max_y << " min_y " << min_y << " max_z " << max_z << " min_z " << min_z << endl;

    //TODO divide by the number of vertices you read in!!!

    cx = cx/faceList.size();
    cy = cy/faceList.size();
    cz = cz/faceList.size();

    //cout << "center " << center.x << " " << center.y << " " << center.z << endl;
    //cout << "scale by " << 1.0/(float)max_extent << endl;
  } else {
    cout << "format is: meshparser filename" << endl;
  }


  glutMainLoop();
}
