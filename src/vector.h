
/*
   Copyright (C) 2000 Nate Miller nkmiller@calpoly.edu

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

   See gpl.txt for more information regarding the GNU General Public License.
   */

#ifndef __VEC3T__
#define __VEC3T__

#include <iostream>
#include <stdio.h>
using namespace std;
#include <math.h>

#ifndef __VEC_T__
#define __VEC_T_
typedef float vec_t;
#endif

struct vec4_t
{
   vec4_t() {v[0] = v[1] = v[2] = v[3] = 0;}
   vec4_t(vec_t px, vec_t py, vec_t pz, vec_t pw) {v[0] = px; v[1] = py; v[2] = pz; v[3] = pw;}
   //vec4_t(vec3_t &v3, vec_t pw) {v[0] = v3.x(); v[1] = v3.y(); v[2] = v3.z(); v[3] = pw;}
   //vec4_t(vec3_t &v3, vec_t pw) {v[0] = v3.v[0]; v[1] = v3.v[1]; v[2] = v3.v[2]; v[3] = pw;}
   
   vec_t x(void) {return v[0];}
   vec_t y(void) {return v[1];}
   vec_t z(void) {return v[2];}
   vec_t w(void) {return v[3];}
   void x(vec_t nx) {v[0] = nx;}
   void y(vec_t ny) {v[1] = ny;}
   void z(vec_t nz) {v[2] = nz;}
   void w(vec_t nw) {v[2] = nw;}

   vec_t v[4];
};

struct vec3_t
{
   vec3_t() {v[0] = v[1] = v[2] = 0;}
   vec3_t(vec_t px, vec_t py, vec_t pz) {v[0] = px; v[1] = py; v[2] = pz;}
   vec3_t(vec3_t &pVec) {v[0] = pVec.v[0]; v[1] = pVec.v[1]; v[2] = pVec.v[2];}
   vec3_t(vec_t *pVec) {v[0] = pVec[0]; v[1] = pVec[1]; v[2] = pVec[2];}

   vec3_t operator=(vec3_t &pVec)
   {return vec3_t(v[0] = pVec.v[0], v[1] = pVec.v[1], v[2] = pVec.v[2]);}
   vec3_t operator=(vec_t *ptr)
   {return vec3_t(v[0] = ptr[0], v[1] = ptr[1], v[2] = ptr[2]);}
   int operator==(vec3_t &pVec)
   {return (v[0] == pVec.v[0] && v[1] == pVec.v[1] && v[2] == pVec.v[2]);}
   int operator==(float *pVec)
   {return (v[0] == pVec[0] && v[1] == pVec[1] && v[2] == pVec[2]);}
   inline int operator!=(vec3_t &pVec)
   {return !(pVec == (*this));}
   inline int operator!=(float *pVec)
   {return !(pVec == (*this));}

   vec3_t operator+=(vec3_t &pVec);
   vec3_t operator-=(vec3_t &pVec);
   vec3_t operator*=(vec3_t &pVec);
   vec3_t operator*=(vec_t val);
   vec3_t operator/=(vec3_t &pVec);
   vec3_t operator/=(vec_t val);

   vec3_t operator+(vec3_t &pVec)
   {return vec3_t(v[0] + pVec.v[0], v[1] + pVec.v[1], v[2] + pVec.v[2]);}
   vec3_t operator-(vec3_t &pVec)
   {return vec3_t(v[0] - pVec.v[0], v[1] - pVec.v[1], v[2] - pVec.v[2]);}
   vec3_t operator*(vec3_t &pVec)
   {return vec3_t(v[0] * pVec.v[0], v[1] * pVec.v[1], v[2] * pVec.v[2]);}
   vec3_t operator*(vec_t val)
   {return vec3_t(v[0] * val, v[1] * val, v[2] * val);}
   vec3_t operator/(vec3_t &pVec)
   {return vec3_t(v[0] / pVec.v[0], v[1] / pVec.v[1], v[2] / pVec.v[2]);}
   vec3_t operator/(vec_t val)
   {return vec3_t(v[0] / val, v[1] / val, v[2] / val);}

   void clear(void) {v[0] = v[1] = v[2] = 0;}
   void normalize(void);
   vec_t length(void);
   vec_t dot(vec3_t &pVec) {return v[0] * pVec.v[0] + v[1] * pVec.v[1] + v[2] * pVec.v[2];}
   void cross(vec3_t &p, vec3_t &q);

   void set(float x, float y, float z) {v[0] = x; v[1] = y; v[2] = z;}

   vec_t x(void) {return v[0];}
   vec_t y(void) {return v[1];}
   vec_t z(void) {return v[2];}
   void x(vec_t nx) {v[0] = nx;}
   void y(vec_t ny) {v[1] = ny;}
   void z(vec_t nz) {v[2] = nz;}

   const vec_t &operator[](int ndx) const {return v[ndx];}
   vec_t &operator[](int ndx) {return v[ndx];}
   operator vec_t*(void) {return v;}

   void clamp(int min, int max);

   void rotateX(float amnt);
   void rotateY(float amnt);
   void rotateZ(float amnt);

   void print();

   friend ostream &operator<<(ostream &os, vec3_t &vec)
   {os << vec.v[0] << " " << vec.v[1] << " " << vec.v[2]; return os;}

   //protected:
   vec_t v[3];
};

inline vec3_t vec3_t::operator+=(vec3_t &pVec)
{
   vec3_t ret;

   ret = *this = *this + pVec;

   return ret;
}

inline vec3_t vec3_t::operator-=(vec3_t &pVec)
{
   vec3_t ret;

   ret = *this = *this - pVec;

   return ret;
}

inline vec3_t vec3_t::operator*=(vec3_t &pVec)
{
   vec3_t ret;

   ret = *this = *this * pVec;

   return ret;
}

inline vec3_t vec3_t::operator*=(vec_t val)
{
   vec3_t ret;

   ret = *this = *this * val;

   return ret;
}

inline vec3_t vec3_t::operator/=(vec3_t &pVec)
{
   vec3_t ret;

   ret = *this = *this / pVec;

   return ret;
}

inline vec3_t vec3_t::operator/=(vec_t val)
{
   vec3_t ret;

   ret = *this = *this / val;

   return ret;
}

inline void vec3_t::normalize(void)
{
   vec_t len0, len1 = 0;

   len0 = length();

   if (len0 == 0)
      return;

   len1 = 1.0f / len0;

   v[0] *= len1;
   v[1] *= len1;
   v[2] *= len1;
}

inline float vec3_t::length(void)
{
   double length = 0;

   length = (v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]);

   return (vec_t) sqrt(length);
}

inline void vec3_t::cross(vec3_t &p, vec3_t &q)
{
   v[0] = p.v[1] * q.v[2] - p.v[2] * q.v[1];
   v[1] = p.v[2] * q.v[0] - p.v[0] * q.v[2];
   v[2] = p.v[0] * q.v[1] - p.v[1] * q.v[0];
}

inline void vec3_t::clamp(int min, int max)
{
   if (v[0] > max || v[0] < min)
      v[0] = 0;

   if (v[1] > max || v[1] < min)
      v[1] = 0;

   if (v[2] > max || v[2] < min)
      v[2] = 0;
}

inline void vec3_t::rotateX(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float y = v[1];
   float z = v[2];

   v[1] = (y * c) - (z * s);
   v[2] = (y * s) + (z * c);
}

inline void vec3_t::rotateY(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float x = v[0];
   float z = v[2];

   v[0] = (x * c) + (z * s);
   v[2] = (z * c) - (x * s);
}

inline void vec3_t::rotateZ(float amnt)
{
   float s = (float)sin(amnt);
   float c = (float)cos(amnt);
   float x = v[0];
   float y = v[1];

   v[0] = (x * c) - (y * s);
   v[1] = (y * c) + (x * s);
}

inline void vec3_t::print()
{
   printf("<%f, %f, %f>\n", v[0], v[1], v[2]);
}
#endif
