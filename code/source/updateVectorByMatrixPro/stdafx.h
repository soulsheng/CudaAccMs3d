// stdafx.h : ��׼ϵͳ�����ļ��İ����ļ���
// ���Ǿ���ʹ�õ��������ĵ�
// �ض�����Ŀ�İ����ļ�
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>


#define    SCALE_CLASS    2 //1  2  3	4	5	6	7	8	9 10
#define    BASE_SIZE     1
#define    JOINT_SIZE    ( 1 << (SCALE_CLASS*2)  )
// ���ݹ�ģ	(M)= JOINT_SIZE * BASE_SIZE * sizeof()	
// 0.002   0.008   0.03	0.13	0.5	2	8	32	128	512	2048

// TODO: �ڴ˴����ó�����Ҫ������ͷ�ļ�
