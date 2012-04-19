#ifndef __FACE_H
#define __FACE_H
#include "point.h"

class Face {
  public:
    Face() {};
    Face(Point p1, Point p2, Point p3);
    ~Face() {};
    void draw();
    void update(Point p1, Point p2, Point p3);
    void print();
  protected:
    Point _v1;
    Point _v2;
    Point _v3;
};

#endif
