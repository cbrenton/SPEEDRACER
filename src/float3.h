#ifndef _FLOAT3_H
#define _FLOAT3_H

#include <stdio.h>

struct float3
{
   float v[3];

   float3()
   {
      v[0] = 0.f;
      v[1] = 0.f;
      v[2] = 0.f;
   }

   float3(int x, int y, int z)
   {
      v[0] = (float)x;
      v[1] = (float)y;
      v[2] = (float)z;
   }

   float3(float x, float y, float z)
   {
      v[0] = x;
      v[1] = y;
      v[2] = z;
   }

   inline float x() {return v[0];}

   inline float y() {return v[1];}

   inline float z() {return v[2];}

   inline const float3 operator -(const float3& p)
   {
      return float3(v[0] - p.v[0],
            v[1] - p.v[1],
            v[2] - p.v[2]);
   }

   inline float dot(float3 &p)
   {
      return v[0] * p.v[0] + v[1] * p.v[1] + v[2] * p.v[2];
   }

   inline float3 cross(float3 &p)
   {
      float3 ret;
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
      float l = (float)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      if (l != 0){
         v[0] = v[0]/l;
         v[1] = v[1]/l;
         v[2] = v[2]/l;
      }
   }
};

#endif
