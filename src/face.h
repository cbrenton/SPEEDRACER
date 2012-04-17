#ifndef __POINT_H
#define __POINT_H
#include "point.h"
#endif

class Face {
  public:
    Face() {};
    Face(Point p1, Point p2, Point p3);
    ~Face() {};
    void draw();
    void update(Point p1, Point p2, Point p3);
    void print();
    /*
    void addV1(&Point p1);
    void addV2(&Point p2);
    void addV3(&Point p3);
    */
  protected:
    Point _v1;
    Point _v2;
    Point _v3;
};
