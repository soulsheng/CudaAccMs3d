// updateVectorByMatrixPro.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


#include "Joint.h"
#include "../common/stopwatch_win.h"

// 数据定义
Joints		_joints;//关节矩阵

// 数据初始化：坐标、矩阵
void initialize(int joint_size);

// 数据销毁：坐标、矩阵
void unInitialize();

int _tmain(int argc, _TCHAR* argv[])
{
	
	int nRepeatPerSecond = 0;// 每秒重复次数，表示时间效率
	
	StopWatchWin timer;
	
	{		
		// 数据初始化：坐标、矩阵
		initialize(JOINT_SIZE*BASE_SIZE);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			cudaMemcpy( _joints.pMatrixDevice, _joints.pMatrix, sizeof(Matrix) * JOINT_SIZE*BASE_SIZE, cudaMemcpyHostToDevice );
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();

		// 数据销毁：坐标、矩阵
		unInitialize();

		// 查看时间效率
		printf("%d: F=%d, T=%.1f us\n", SCALE_CLASS, nRepeatPerSecond/10, 10000000.0f/nRepeatPerSecond);
	}
	
	// 输出结果：绘制坐标，按照点、线、面的形式
	// ...省略

	return 0;
}

// 数据初始化：坐标、矩阵
void initialize(int joint_size)
{
	_joints.initialize( joint_size );
}

// 数据销毁：坐标、矩阵
void unInitialize()
{
	_joints.unInitialize();
}
