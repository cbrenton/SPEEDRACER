#ifndef _TRI_T_H
#define _TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "vec3.h"

#define EPSILON 0.001f

struct tri_t
{
   point_t **pt;

   tri_t()
   {
      pt = new point_t*[3];
   }

   tri_t(point_t *_p1, point_t *_p2, point_t *_p3)
   {
      pt = new point_t*[3];
      pt[0] = _p1;
      pt[1] = _p2;
      pt[2] = _p3;
   }

   ~tri_t()
   {
      delete [] pt;
   }

   inline void w2p(int w, int h)
   {
      for (int i = 0; i < 3; i++)
      {
         pt[i]->w2p(w, h);
      }
   }

   bool hit(int x, int y, vec_t *t)
   {
      bool hit = true;

      vec_t bBeta, bGamma, bT;

      vec3 pix (x, y, 0);
      vec3 screenPt[3];
      for (int i = 0; i < 3; i++)
      {
         screenPt[i] = pt[i]->toF3Screen();
      }

      mat_t A (screenPt[0].x(), screenPt[1].x(), screenPt[2].x(),
            screenPt[0].y(), screenPt[1].y(), screenPt[2].y(),
            1.f, 1.f, 1.f);

      vec_t detA = A.det();
      if (detA == 0)
      {
         return false;
      }

      mat_t baryT (pix.x(), screenPt[1].x(), screenPt[2].x(),
            pix.y(), screenPt[1].y(), screenPt[2].y(),
            1.f, 1.f, 1.f);

      bT = baryT.det() / detA;

      if (bT < 0)
      {
         hit = false;
      }
      else
      {
         mat_t baryGamma (screenPt[0].x(), pix.x(), screenPt[2].x(),
               screenPt[0].y(), pix.y(), screenPt[2].y(),
               1.f, 1.f, 1.f);

         bGamma = baryGamma.det() / detA;

         if (bGamma < 0 || bGamma > 1)
         {
            hit = false;
         }
         else
         {
            mat_t baryBeta (screenPt[0].x(), screenPt[1].x(), pix.x(),
                  screenPt[0].y(), screenPt[1].y(), pix.y(),
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
         *t = bT * pt[0]->coords.v[2] + bBeta * pt[1]->coords.v[2] + bGamma *
            pt[2]->coords.v[2];
      }
      return hit;
   }
};

#endif
