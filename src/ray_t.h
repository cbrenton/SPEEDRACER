/**
 * Represents a ray with both origin and direction.
 * @author Chris Brenton
 * @date 06/20/2011
 */

#ifndef __RAY_T_H
#define __RAY_T_H

#include "vector.h"

struct ray_t
{
   // The origin of the ray.
   vec3_t point;

   // The direction of the ray.
   vec3_t dir;

};
#endif
