#ifndef _TRI_T_H
#define _TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "vector.h"

#define EPSILON 0.001f

struct tri_t
{
   point_t **pt;
   vec3_t *normal;
   int *extents;
   bool perVert;

   tri_t()
   {
      perVert = false;
      pt = new point_t*[3];
      extents = new int[4];
      if (perVert)
         normal = new vec3_t();
      else
         normal = new vec3_t[3];
      genNormal();
   }

   tri_t(point_t *_p1, point_t *_p2, point_t *_p3)
   {
      perVert = false;
      pt = new point_t*[3];
      extents = new int[4];
      pt[0] = _p1;
      pt[1] = _p2;
      pt[2] = _p3;
      if (perVert)
         normal = new vec3_t();
      else
         normal = new vec3_t[3];
      genNormal();
   }

   ~tri_t()
   {
      delete [] pt;
      if (perVert)
         delete normal;
      else
         delete [] normal;
      delete [] extents;
   }

   void genExtents()
   {
      int minX = pt[0]->pX;
      int maxX = pt[0]->pX;
      int minY = pt[0]->pY;
      int maxY = pt[0]->pY;
      for (int i = 0; i < 3; i++)
      {
         if (pt[i]->pX < minX)
            minX = pt[i]->pX;
         if (pt[i]->pX > maxX)
            maxX = pt[i]->pX;
         if (pt[i]->pY < minY)
            minY = pt[i]->pY;
         if (pt[i]->pY > maxY)
            maxY = pt[i]->pY;
      }
      extents[0] = minX;
      extents[1] = maxX;
      extents[2] = minY;
      extents[3] = maxY;
   }

   void genNormal()
   {
      if (!perVert)
      {
         // Calculate the normal.
         vec3_t ab = pt[0]->toF3World();
         vec3_t ab2 = pt[1]->toF3World();
         ab -= ab2;
         vec3_t ac = pt[0]->toF3World();
         vec3_t ac2 = pt[2]->toF3World();
         ac -= ac2;
         normal->cross(ab, ac);
         normal->normalize();
      }
      else
      {
         normal[0] = vec3_t(1.f, 0.f, 0.f);
         normal[1] = vec3_t(0.f, 1.f, 0.f);
         normal[2] = vec3_t(0.f, 0.f, 1.f);
      }
   }

   void w2p(int w, int h)
   {
      for (int i = 0; i < 3; i++)
      {
         pt[i]->w2p(w, h);
      }
   }

   bool hit(int x, int y, vec_t *t, vec3_t *bary = NULL)
   {
#ifndef _CUDA
      if (x < extents[0] || x > extents[1] ||
            y < extents[2] || y > extents[3])
         return false;
#endif
      bool hit = true;

      vec_t bBeta, bGamma, bT;

      vec3_t pix ((vec_t)x, (vec_t)y, 0);
      vec3_t screenPt[3];
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
