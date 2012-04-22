#ifndef _MAT_T_H
#define _MAT_T_H

struct mat_t
{
   float data[3][3];

   mat_t(int n1, int n2, int n3,
         int n4, int n5, int n6,
         int n7, int n8, int n9)
   {
      data[0][0] = (float)n1;
      data[0][1] = (float)n2;
      data[0][2] = (float)n3;
      data[1][0] = (float)n4;
      data[1][1] = (float)n5;
      data[1][2] = (float)n6;
      data[2][0] = (float)n7;
      data[2][1] = (float)n8;
      data[2][2] = (float)n9;
   }
   
   mat_t(float n1, float n2, float n3,
         float n4, float n5, float n6,
         float n7, float n8, float n9)
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

   inline float det()
   {
      return data[0][0] * data[1][1] * data[2][2] + data[0][1] * data[1][2] *
         data[2][0] + data[0][2] * data[1][0] * data[2][1] - data[0][2] *
         data[1][1] * data[2][0] - data[0][0] * data[1][2] * data[2][1] -
         data[0][1] * data[1][0] * data[2][2];
   }
};

#endif
