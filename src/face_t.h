#ifndef __FACE_T_H
#define __FACE_T_H

#include "point_t.h"

struct face_t
{
   point_t *p1, *p2, *p3;

   face_t() {};

   face_t(point_t *_p1, point_t *_p2, point_t *_p3) :
      p1(_p1), p2(_p2), p3(_p3)
   {
   }
};

#endif
