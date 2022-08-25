/* This file is intentionally empty.  You should fill it in with your
   solution to the programming exercise. */

#include <stdio.h>

#include "util.h"
#include "slp.h"
#include "prog1.h"
#include "maxargs.h"

int main()
{
   A_stm a_stm = prog();
   int res = maxargs(a_stm);
   printf("The maximum number of arguments of print is %d.\n", res);
   return 0;
}