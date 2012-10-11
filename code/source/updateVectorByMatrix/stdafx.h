// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#define    MEGA_SIZE     (1<<20)  // Mega, or million
#define    PROBLEM_SCALE 1 // 1, 4, 16, 64, 256
#define    PROBLEM_SIZE  ( MEGA_SIZE * PROBLEM_SCALE )  // n Mega elements
#define    JOINT_SIZE    100


// TODO: 在此处引用程序需要的其他头文件
