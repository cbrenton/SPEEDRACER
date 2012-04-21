#ifndef _FLOAT3_H
#define _FLOAT3_H

struct float3
{
   float v[3];

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

   const float3 operator -(const float3& other)
   {
      return float3(v[0] - other.v[0],
            v[1] - other.v[1],
            v[2] - other.v[2]);
   }
   
   float dot(float3 &other) {
      return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2];
   }
};

#endif
