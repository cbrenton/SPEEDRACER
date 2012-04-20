#ifndef __TRI_T_H
#define __TRI_T_H

#include "mat_t.h"
#include "ray_t.h"
#include "point_t.h"

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

   int triangle_hit(const ray_t & ray, float *t)
   {
      float result = -1;
      float bBeta, bGamma, bT;

      /*
      Matrix3f A;
      A <<
         c1.x()-c2.x(), c1.x()-c3.x(), ray.dir.x(),
         c1.y()-c2.y(), c1.y()-c3.y(), ray.dir.y(),
         c1.z()-c2.z(), c1.z()-c3.z(), ray.dir.z();
      float detA = A.determinant();

      Matrix3f baryT;
      baryT <<
         c1.x()-c2.x(), c1.x()-c3.x(), c1.x()-ray.point.x(),
         c1.y()-c2.y(), c1.y()-c3.y(), c1.y()-ray.point.y(),
         c1.z()-c2.z(), c1.z()-c3.z(), c1.z()-ray.point.z();

      bT = baryT.determinant() / detA;
      */

      if (bT < 0)
      {
         result = 0;
      }
      else
      {
         /*
         Matrix3f baryGamma;
         baryGamma <<
            c1.x()-c2.x(), c1.x()-ray.point.x(), ray.dir.x(),
            c1.y()-c2.y(), c1.y()-ray.point.y(), ray.dir.y(),
            c1.z()-c2.z(), c1.z()-ray.point.z(), ray.dir.z();

         bGamma = baryGamma.determinant() / detA;
         */

         if (bGamma < 0 || bGamma > 1)
         {
            result = 0;
         }
         else
         {
            /*
            Matrix3f baryBeta;
            baryBeta <<
               c1.x()-ray.point.x(), c1.x()-c3.x(), ray.dir.x(),
               c1.y()-ray.point.y(), c1.y()-c3.y(), ray.dir.y(),
               c1.z()-ray.point.z(), c1.z()-c3.z(), ray.dir.z();

            bBeta = baryBeta.determinant() / detA;
            */

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
   }
};

#endif
