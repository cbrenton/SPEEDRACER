#ifndef __POINT_T_H
#define __POINT_T_H

#include <stdio.h>

struct point_t
{
   int index; // The index of this point.
   float x, y, z; // The coordinates of the point in 3D space.

   point_t() {};

   point_t(int _index, float _x, float _y, float _z) :
      index(_index), x(_x), y(_y), z(_z)
   {
   }

   inline bool isNum(int check)
   {
      return (index == check);
   }
   
   inline void print()
   {
      printf("\t%d %f %f %f\n", index, x, y, z);
   }
};

#endif
