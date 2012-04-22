#ifndef _vec3_H
#define _vec3_H

#include <stdio.h>

#ifdef _USEDBL
typedef double vec_t;
#else
typedef float vec_t;
#endif

struct vec3
{
   vec_t v[3];

   vec3()
   {
      v[0] = 0.f;
      v[1] = 0.f;
      v[2] = 0.f;
   }

   vec3(int x, int y, int z)
   {
      v[0] = (vec_t)x;
      v[1] = (vec_t)y;
      v[2] = (vec_t)z;
   }

   vec3(vec_t x, vec_t y, vec_t z)
   {
      v[0] = x;
      v[1] = y;
      v[2] = z;
   }

   inline vec_t x() {return v[0];}

   inline vec_t y() {return v[1];}

   inline vec_t z() {return v[2];}

   inline const vec3 operator -(const vec3& p)
   {
      return vec3(v[0] - p.v[0],
            v[1] - p.v[1],
            v[2] - p.v[2]);
   }

   inline vec_t dot(vec3 &p)
   {
      return v[0] * p.v[0] + v[1] * p.v[1] + v[2] * p.v[2];
   }

   inline vec3 cross(vec3 &p)
   {
      vec3 ret;
      ret.v[0] = v[1] * p.v[2] - v[2] * p.v[1];
      ret.v[1] = v[2] * p.v[0] - v[0] * p.v[2];
      ret.v[2] = v[0] * p.v[1] - v[1] * p.v[0];
      return ret;
   }

   inline void print()
   {
      printf("<%f %f %f>\n", v[0], v[1], v[2]);
   }

   inline void normalize()
   {
      vec_t l = (vec_t)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      if (l != 0){
         v[0] = v[0]/l;
         v[1] = v[1]/l;
         v[2] = v[2]/l;
      }
   }
};

#endif
