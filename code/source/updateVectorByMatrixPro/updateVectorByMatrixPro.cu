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
int iClass=6; // 问题规模最大值，16M/512M显存、32M/1G显存

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
			globalMemoryUpdate( &_joints );
			
			dim3 nBlocksPerGrid( 64 ); // 块的数目
			dim3 nThreadsPerBlock( 256 ); // 单块包含线程的数目

#if USE_ELEMENT_SINGLE
			nBlocksPerGrid.y = (PROBLEM_SIZE+nThreadsPerBlock.x - 1)/(nThreadsPerBlock.x * nBlocksPerGrid.x);
#endif

			// 执行运算：坐标矩阵变换
#if SEPERATE_STRUCT

#if USE_SHARED
			int sizeMatrixShared = sizeof(float4) * _joints.nSize * 3 ;
			updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock, sizeMatrixShared>>>
				(_vertexesStatic.pVertexDevice, _vertexesStatic.nSize, _joints.pMatrixDevice[0], _vertexesDynamic.pVertexDevice,
				_joints.pMatrixDevice[1], _joints.pMatrixDevice[2], _joints.nSize );
#else
			updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock>>>
				(_vertexesStatic.pVertexDevice, _vertexesStatic.nSize, _joints.pMatrixDevice[0], _vertexesDynamic.pVertexDevice,
				_joints.pMatrixDevice[1], _joints.pMatrixDevice[2] );
#endif

#else

			updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock>>>
				(_vertexesStatic.pVertexDevice, _vertexesDynamic.nSize, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice, _joints.nSize ,
				_joints.pMatrixDevicePrevious);

#endif

			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();

		// 验证GPU运算的正确性，是否和CPU运算结果一致
		bool bResult = false;

		// 获取CPU运算结果
		updateVectorByMatrixGold(_vertexesStatic.pVertex, _vertexesDynamic.nSize, &_joints, _vertexesDynamic.pVertex);

		// 获取GPU运算结果
		Vector4 *pVertex = new Vector4[_vertexesDynamic.nSize];
		cudaMemcpy( pVertex, _vertexesDynamic.pVertexDevice, sizeof(Vector4) * _vertexesDynamic.nSize, cudaMemcpyDeviceToHost );
		
		// 比较结果
		bResult = equalVector( _vertexesDynamic.pVertex , _vertexesDynamic.nSize, pVertex );
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
	_joints.initialize( joint_size );
#if USE_MEMORY_BUY_TIME
	_vertexesStatic.initialize( problem_size, joint_size );
#endif
	_vertexesDynamic.initialize( problem_size, joint_size );
}

// 数据销毁：坐标、矩阵
void unInitialize()
{
	_joints.unInitialize();
#if USE_MEMORY_BUY_TIME
	_vertexesStatic.unInitialize();
#endif
	_vertexesDynamic.unInitialize();
}
