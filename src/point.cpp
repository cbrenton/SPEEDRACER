#include <stdio.h>
#include "point.h"

Point::Point()
{
   _num = 0;
   _x = 0;
   _y = 0;
   _z = 0;
}

Point::Point(int num, float x, float y, float z) :
   _num(num), _x(x), _y(y), _z(z) {

}

void Point::draw() {
   //glVertex3f(_x, _y, _z);
}

void Point::print() {
   printf("\t%d %f %f %f\n", _num, _x, _y, _z);
}

bool Point::isNum(int check) {
   return (_num == check);
}

void Point::update(int num, float x, float y, float z) {
   _num = num;
   _x = x;
   _y = y;
   _z = z;
}
