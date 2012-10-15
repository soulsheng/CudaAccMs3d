// updateVectorByMatrixPro.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "updateVectorByMatrixPro.cuh"
#include "updateVectorByMatrix.h"

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=6; // 问题规模最大值，16M/1G显存、32M/2G显存

// 数据定义
Vertexes  _vertexesStatic;//静态顶点坐标
Vertexes  _vertexesDynamic;//动态顶点坐标
Joints		_joints;//关节矩阵

// 数据初始化：坐标、矩阵
void initialize(int problem_size, int joint_size);

// 数据销毁：坐标、矩阵
void unInitialize();

int _tmain(int argc, _TCHAR* argv[])
{
	
	int nRepeatPerSecond = 0;// 每秒重复次数，表示时间效率
	
	StopWatchWin timer;
	
	{
		// 问题规模档次，7档，64K至256M，4倍递增
		PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;
		
		// 数据初始化：坐标、矩阵
		initialize(PROBLEM_SIZE, JOINT_SIZE);
		
		timer.start();

		while ( timer.getTime() < 10000  )
		{
#if USE_CONSTANT
			// 改变矩阵存储方式，从全局显存改为常量显存，后者具有缓存
			constantMemoryUpdate( _joints.pMatrix, JOINT_SIZE );
#else
			cudaMemcpy( _joints.pMatrixDevice, _joints.pMatrix, sizeof(Matrix) * JOINT_SIZE, cudaMemcpyHostToDevice );
#endif
			// 执行运算：坐标矩阵变换
			updateVectorByMatrix<<<64, 256>>>(_vertexesStatic.pVertexDevice, PROBLEM_SIZE, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice);
			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();

		// 验证GPU运算的正确性，是否和CPU运算结果一致
		bool bResult = false;

		// 获取CPU运算结果
		updateVectorByMatrixGold(_vertexesStatic.pVertex, PROBLEM_SIZE, _joints.pMatrix, _vertexesDynamic.pVertex);

		// 获取GPU运算结果
		Vector4 *pVertex = new Vector4[PROBLEM_SIZE];
		cudaMemcpy( pVertex, _vertexesDynamic.pVertexDevice, sizeof(Vector4) * PROBLEM_SIZE, cudaMemcpyDeviceToHost );
		
		// 比较结果
		bResult = equalVector( _vertexesDynamic.pVertex , PROBLEM_SIZE, pVertex );
		printf("%s\n", bResult?"Right":"Wrong");

		// 数据销毁：坐标、矩阵
		unInitialize();

		// 查看时间效率
		printf("%d: F=%d, T=%.2f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
	}
	
	// 输出结果：绘制坐标，按照点、线、面的形式
	// ...省略

	return 0;
}

// 数据初始化：坐标、矩阵
void initialize(int problem_size, int joint_size)
{
	_joints.initialize( JOINT_SIZE );
	_vertexesStatic.initialize( PROBLEM_SIZE, JOINT_SIZE );
	_vertexesDynamic.initialize( PROBLEM_SIZE, JOINT_SIZE );
}

// 数据销毁：坐标、矩阵
void unInitialize()
{
	_joints.unInitialize();
	_vertexesStatic.unInitialize();
	_vertexesDynamic.unInitialize();
}
