#ifndef __POINT_T_H
#define __POINT_T_H

#include <stdio.h>

struct point_t
{
   int index; // The index of this point.
   float x, y, z; // The coordinates of the point in 3D space.
};

void printPt(point_t pt)
{
   printf("\t%d %f %f %f\n", pt.index, pt.x, pt.y, pt.z);
}

#endif
