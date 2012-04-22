#ifndef _TRI_T_H
#define _TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "vec3.h"

#define EPSILON 0.001f

struct tri_t
{
   point_t **pt;
   vec3 *normal;
   bool perVert;

   tri_t()
   {
      perVert = false;
      pt = new point_t*[3];
      if (perVert)
         normal = new vec3();
      else
         normal = new vec3[3];
      genNormal();
   }

   tri_t(point_t *_p1, point_t *_p2, point_t *_p3)
   {
      perVert = false;
      pt = new point_t*[3];
      pt[0] = _p1;
      pt[1] = _p2;
      pt[2] = _p3;
      if (perVert)
         normal = new vec3();
      else
         normal = new vec3[3];
      genNormal();
   }

   ~tri_t()
   {
      delete [] pt;
   }

   inline void genNormal()
   {
      if (!perVert)
      {
         // Calculate the normal.
         vec3 ab = pt[0]->toF3World() - pt[1]->toF3World();
         vec3 ac = pt[0]->toF3World() - pt[2]->toF3World();
         *normal = ab.cross(ac);
         normal->normalize();
      }
      else
      {
         normal[0] = vec3(1.f, 0.f, 0.f);
         normal[1] = vec3(0.f, 1.f, 0.f);
         normal[2] = vec3(0.f, 0.f, 1.f);
      }
   }

   inline vec3 * getNormal(vec3 *bary)
   {
      vec3 *ret = new vec3();
      for (int i = 0; i < 3; i++)
      {
         ret->v[i] = normal->v[i] * bary->v[i];
      }
      return ret;
   }

   inline void w2p(int w, int h)
   {
      for (int i = 0; i < 3; i++)
      {
         pt[i]->w2p(w, h);
      }
   }

   bool hit(int x, int y, vec_t *t, vec3 *bary = NULL)
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
         if (bary)
         {
            bary->v[0] = bT;
            bary->v[1] = bBeta;
            bary->v[2] = bGamma;
         }
      }
      return hit;
   }
};

#endif
