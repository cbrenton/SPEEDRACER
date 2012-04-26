#ifndef _TRI_T_H
#define _TRI_T_H

#include "mat_t.h"
#include "point_t.h"
#include "vector.h"
#include <vector>
#include <climits>

#define EPSILON 0.001f

struct tri_t
{
   point_t *ptList;
   int numPts;
   int pt[3];
   vec3_t normal;
   int extents[4];

   void debug()
{
/*if (extents[0] != 0 || extents[1] != 0 ||
extents[2] != 0 || extents[3] != 0)
{
*/
printf("pt: [%d, %d, %d]\n", pt[0], pt[1], pt[2]);
for (int i = 0; i < 4; i++)
{
printf("%d,", extents[i]);
}
printf("\n");
//}
}

   tri_t()
   {
   }

   tri_t(int _p1, int _p2, int _p3, point_t *list, int listSize) :
      ptList(list), numPts(listSize)
   {
      pt[0] = _p1;
      pt[1] = _p2;
      pt[2] = _p3;
      extents[0] = INT_MAX;
      extents[1] = -INT_MAX;
      extents[2] = INT_MAX;
      extents[3] = -INT_MAX;
   }

   tri_t(const tri_t& tri)
   {
      ptList = tri.ptList;
      numPts = tri.numPts;
      pt[0] = tri.pt[0];
      pt[1] = tri.pt[1];
      pt[2] = tri.pt[2];
      normal[0] = tri.normal[0];
      normal[1] = tri.normal[1];
      normal[2] = tri.normal[2];
      extents[0] = tri.extents[0];
      extents[1] = tri.extents[1];
      extents[2] = tri.extents[2];
      extents[3] = tri.extents[3];
   }

   ~tri_t()
   {
   }

   point_t * getPt(int index)
   {
      if (index > numPts || index < 0)
      {
         fprintf(stderr, "tri_t.getPt(): index must be a valid array index.\n");
         exit(EXIT_FAILURE);
      }
      return &ptList[index];
   }

   void genExtents()
   {
      for (int i = 0; i < 3; i++)
      {
         if (getPt(pt[i])->pX <extents[0])
            extents[0] = getPt(pt[i])->pX;
         if (getPt(pt[i])->pX >extents[1])
            extents[1] = getPt(pt[i])->pX;
         if (getPt(pt[i])->pY <extents[2])
            extents[2] = getPt(pt[i])->pY;
         if (getPt(pt[i])->pY >extents[3])
            extents[3] = getPt(pt[i])->pY;
      }
   }

   void genNormal()
   {
      // Calculate the normal.
      vec3_t ab = getPt(pt[0])->toF3World();
      vec3_t ab2 = getPt(pt[1])->toF3World();
      ab -= ab2;
      vec3_t ac = getPt(pt[0])->toF3World();
      vec3_t ac2 = getPt(pt[2])->toF3World();
      ac -= ac2;
      normal.cross(ab, ac);
      normal.normalize();
   }

#ifndef _CUDA
   bool hit(int x, int y, vec_t *t, vec3_t *bary = NULL)
   {
      if (x < extents[0] || x > extents[1] ||
            y < extents[2] || y > extents[3])
         return false;
      bool hit = true;

      vec_t bBeta, bGamma, bT;

      vec3_t pix ((vec_t)x, (vec_t)y, 0);
      vec3_t screenPt[3];
      for (int i = 0; i < 3; i++)
      {
         screenPt[i] = getPt(pt[i])->toF3Screen();
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
         *t = bT * getPt(pt[0])->coords.v[2] + bBeta * getPt(pt[1])->coords.v[2] + bGamma *
            getPt(pt[2])->coords.v[2];
         if (bary)
         {
            bary->v[0] = bT;
            bary->v[1] = bBeta;
            bary->v[2] = bGamma;
         }
      }
      return hit;
   }
#endif
};

#endif
