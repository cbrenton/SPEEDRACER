#ifndef _MAT_T_H
#define _MAT_T_H

#include "vector.h"

struct mat_t
{
   vec_t data[3][3];

   mat_t(int n1, int n2, int n3,
         int n4, int n5, int n6,
         int n7, int n8, int n9)
   {
      data[0][0] = (vec_t)n1;
      data[0][1] = (vec_t)n2;
      data[0][2] = (vec_t)n3;
      data[1][0] = (vec_t)n4;
      data[1][1] = (vec_t)n5;
      data[1][2] = (vec_t)n6;
      data[2][0] = (vec_t)n7;
      data[2][1] = (vec_t)n8;
      data[2][2] = (vec_t)n9;
   }
   
   mat_t(vec_t n1, vec_t n2, vec_t n3,
         vec_t n4, vec_t n5, vec_t n6,
         vec_t n7, vec_t n8, vec_t n9)
   {
      data[0][0] = n1;
      data[0][1] = n2;
      data[0][2] = n3;
      data[1][0] = n4;
      data[1][1] = n5;
      data[1][2] = n6;
      data[2][0] = n7;
      data[2][1] = n8;
      data[2][2] = n9;
   }

   inline vec_t det()
   {
      return data[0][0] * data[1][1] * data[2][2] + data[0][1] * data[1][2] *
         data[2][0] + data[0][2] * data[1][0] * data[2][1] - data[0][2] *
         data[1][1] * data[2][0] - data[0][0] * data[1][2] * data[2][1] -
         data[0][1] * data[1][0] * data[2][2];
   }
};

#endif
