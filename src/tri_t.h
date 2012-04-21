#ifndef __TRI_T_H
#define __TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "float3.h"

#define EPSILON 0.001f

struct tri_t
{
   point_t *p1, *p2, *p3;

   tri_t() {};

   tri_t(point_t *_p1, point_t *_p2, point_t *_p3) :
      p1(_p1), p2(_p2), p3(_p3)
   {
   }

   inline void w2p(int w, int h)
   {
      p1->w2p(w, h);
      p2->w2p(w, h);
      p3->w2p(w, h);
   }

   int hit(int x, int y, float *t)
   {
      float3 p (x, y, 0);

      float3 v0 = p3->toF3() - p1->toF3();
      float3 v1 = p2->toF3() - p1->toF3();
      float3 v2 = p - p1->toF3();

      float d00 = v0.dot(v0);
      float d01 = v0.dot(v1);
      float d02 = v0.dot(v2);
      float d11 = v1.dot(v1);
      float d12 = v1.dot(v2);

      float invDenom = 1 / (d00 * d11 - d01 * d01);
      float u = (d11 * d02 - d01 * d12) * invDenom;
      float v = (d00 * d12 - d01 * d02) * invDenom;
      return (u >= 0) && (v >= 0) && (u + v < 1);

      //printf("(%d, %d): %d %d 0\n", x, y, p1->pX, p1->pY);
      /*
      float result = -1;

      float bBeta, bGamma, bT;

      mat_t A (p1->pX-p2->pX, p1->pX-p3->pX, x,
         p1->pY-p2->pY, p1->pY-p3->pY, y,
         0, 0, 0);
      float detA = A.det();

      mat_t baryT (p1->pX-p2->pX, p1->pX-p3->pX, p1->pX-x,
         p1->pY-p2->pY, p1->pY-p3->pY, p1->pY-y,
         0, 0, 0);

      bT = baryT.det() / detA;

      if (bT < 0)
      {
         result = 0;
      }
      else
      {
         mat_t baryGamma (p1->pX-p2->pX, p1->pX-x, x,
            p1->pY-p2->pY, p1->pY-y, y,
            0, 0, 0);

         bGamma = baryGamma.det() / detA;

         if (bGamma < 0 || bGamma > 1)
         {
            result = 0;
         }
         else
         {
            mat_t baryBeta (p1->pX-x, p1->pX-p3->pX, x,
               p1->pY-y, p1->pY-p3->pY, y,
               0, 0, 0);

            bBeta = baryBeta.det() / detA;

            if (bBeta < 0 || bBeta > 1 - bGamma)
            {
               result = 0;
            }
         }
      }

      if (result != 0)
      {
         result = bT;
      }
      *t = result;
      if (result > EPSILON)
      {
         // TODO: Record hit data.
         return 1;
      }
      return 0;
      */
   }
};

#endif
