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
   //int pt[3];
   int pt0, pt1, pt2;
   vec3_t normal;
   int extents[4];

   void debug()
{
/*if (extents[0] != 0 || extents[1] != 0 ||
extents[2] != 0 || extents[3] != 0)
{
*/
//printf("pt: [%d, %d, %d]\n", pt[0], pt[1], pt[2]);
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
      pt0 = _p1;
      pt1 = _p2;
      pt2 = _p3;
      extents[0] = INT_MAX;
      extents[1] = -INT_MAX;
      extents[2] = INT_MAX;
      extents[3] = -INT_MAX;
   }

   tri_t(const tri_t& tri)
   {
      ptList = tri.ptList;
      numPts = tri.numPts;
      pt0 = tri.pt0;
      pt1 = tri.pt1;
      pt2 = tri.pt2;
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

   void genExtents(int w, int h)
   {
      /*
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
      */
         if (getPt(pt0)->pX <extents[0])
            extents[0] = getPt(pt0)->pX;
         if (getPt(pt0)->pX >extents[1])
            extents[1] = getPt(pt0)->pX;
         if (getPt(pt0)->pY <extents[2])
            extents[2] = getPt(pt0)->pY;
         if (getPt(pt0)->pY >extents[3])
            extents[3] = getPt(pt0)->pY;


         if (getPt(pt1)->pX <extents[0])
            extents[0] = getPt(pt1)->pX;
         if (getPt(pt1)->pX >extents[1])
            extents[1] = getPt(pt1)->pX;
         if (getPt(pt1)->pY <extents[2])
            extents[2] = getPt(pt1)->pY;
         if (getPt(pt1)->pY >extents[3])
            extents[3] = getPt(pt1)->pY;


         if (getPt(pt2)->pX <extents[0])
            extents[0] = getPt(pt2)->pX;
         if (getPt(pt2)->pX >extents[1])
            extents[1] = getPt(pt2)->pX;
         if (getPt(pt2)->pY <extents[2])
            extents[2] = getPt(pt2)->pY;
         if (getPt(pt2)->pY >extents[3])
            extents[3] = getPt(pt2)->pY;
      extents[0] = min(extents[0], w);
      extents[1] = max(extents[1], 0);
      extents[2] = min(extents[2], h);
      extents[3] = max(extents[3], 0);
   }

   void genNormal()
   {
      // Calculate the normal.
      vec3_t ab = getPt(pt0)->coords;
      vec3_t ab2 = getPt(pt1)->coords;
      ab -= ab2;
      vec3_t ac = getPt(pt0)->coords;
      vec3_t ac2 = getPt(pt2)->coords;
      ac -= ac2;
      normal.cross(ab, ac);
      normal.normalize();
   }
};

#endif
