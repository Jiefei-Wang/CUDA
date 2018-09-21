#pragma once
#ifndef LARGEINDEX
#define LARGEINDEX unsigned long long int
#endif

#ifndef __DEBUG_MODE__
//#define __DEBUG_MODE__ false
#endif

#ifndef __alldev__
#define __alldev__ __host__ __device__
#endif


#ifndef BlockNum
#define BlockNum 100
#define ThreadNum 10
#endif


#include "cuda_runtime.h"
#include "device_launch_parameters.h"