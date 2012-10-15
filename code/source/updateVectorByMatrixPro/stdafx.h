// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>


#define    SCALE_CLASS    8 //1	2	3	4	5	6	7	8	9
#define    BASE_SIZE     625
#define    JOINT_SIZE    (1<<((SCALE_CLASS-1)*2))
// 数据规模	(M)= JOINT_SIZE * BASE_SIZE * sizeof()	
// 0.03	0.13	0.5	2	8	32	128	512	2048

// TODO: 在此处引用程序需要的其他头文件
