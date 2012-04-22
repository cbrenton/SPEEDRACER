#ifndef _TRI_T_H
#define _TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "vec3.h"

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

   bool hit(int x, int y, vec_t *t)
   {
      bool hit = true;

      vec_t bBeta, bGamma, bT;

      vec3 pix (x, y, 0);
      vec3 pt1 = p1->toF3Screen();
      vec3 pt2 = p2->toF3Screen();
      vec3 pt3 = p3->toF3Screen();

      mat_t A (pt1.x(), pt2.x(), pt3.x(),
            pt1.y(), pt2.y(), pt3.y(),
            1.f, 1.f, 1.f);

      vec_t detA = A.det();
      if (detA == 0)
      {
         return false;
      }

      mat_t baryT (pix.x(), pt2.x(), pt3.x(),
            pix.y(), pt2.y(), pt3.y(),
            1.f, 1.f, 1.f);

      bT = baryT.det() / detA;

      if (bT < 0)
      {
         hit = false;
      }
      else
      {
         mat_t baryGamma (pt1.x(), pix.x(), pt3.x(),
               pt1.y(), pix.y(), pt3.y(),
               1.f, 1.f, 1.f);

         bGamma = baryGamma.det() / detA;

         if (bGamma < 0 || bGamma > 1)
         {
            hit = false;
         }
         else
         {
            mat_t baryBeta (pt1.x(), pt2.x(), pix.x(),
                  pt1.y(), pt2.y(), pix.y(),
                  1.f, 1.f, 1.f);

            bBeta = baryBeta.det() / detA;

            if (bBeta < 0 || bBeta > 1 - bGamma)
            {
               hit = false;
            }
         }
      }

      if (hit)
      {
         *t = bT * p1->z + bBeta * p2->z + bGamma * p3->z;
      }
      return hit;
   }
};

#endif
