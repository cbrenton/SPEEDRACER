#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>

#include "face.h"
#include "point.h"

Face::Face(Point p1, Point p2, Point p3):
  _v1(p1), _v2(p2), _v3(p3) {

  }

void Face::draw() {
  //glColor3f(0.0, 1.0, 0.0);
  //glBegin(GL_TRIANGLES);
  _v1.draw();
  _v2.draw();
  _v3.draw();
  //glEnd();
}

void Face::print() {
  _v1.print();
  _v2.print();
  _v3.print();
}

void Face::update(Point p1, Point p2, Point p3) {
  _v1 = p1;
  _v2 = p2;
  _v3 = p3;
}

/*
   void Face::addV1(&Point p1) {
   _v1 = *p1;
   }

   void Face::addV2(&Point p2) {
   _v2 = *p2;
   }

   void Face::addV3(&Point p3) {
   _v3 = *p3;
   }
   */
