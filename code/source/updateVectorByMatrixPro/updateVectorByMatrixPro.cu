// updateVectorByMatrixPro.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "updateVectorByMatrixPro.cuh"
#include "updateVectorByMatrix.h"

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // �����ģ���Σ�8����250K��32M��2������
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// �����ģ, ��ʼ��Ϊ1M����һ����
int iClass=6; // �����ģ���ֵ��16M/512M�Դ桢32M/1G�Դ�

// ���ݶ���
Vertexes  _vertexesStatic;//��̬��������
Vertexes  _vertexesDynamic;//��̬��������
Joints		_joints;//�ؽھ���

// ���ݳ�ʼ�������ꡢ����
void initialize(int problem_size, int joint_size);

// �������٣����ꡢ����
void unInitialize();

int _tmain(int argc, _TCHAR* argv[])
{
	
	int nRepeatPerSecond = 0;// ÿ���ظ���������ʾʱ��Ч��
	
	StopWatchWin timer;
	
	{
		// �����ģ���Σ�7����64K��256M��4������
		PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;
		
		// ���ݳ�ʼ�������ꡢ����
		initialize(PROBLEM_SIZE, JOINT_SIZE);
		
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			globalMemoryUpdate( &_joints );
			
			dim3 nBlocksPerGrid( 64 ); // �����Ŀ
			dim3 nThreadsPerBlock( 256 ); // ��������̵߳���Ŀ

#if USE_ELEMENT_SINGLE
			nBlocksPerGrid.y = (PROBLEM_SIZE+nThreadsPerBlock.x - 1)/(nThreadsPerBlock.x * nBlocksPerGrid.x);
#endif

			// ִ�����㣺�������任
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

		// ��֤GPU�������ȷ�ԣ��Ƿ��CPU������һ��
		bool bResult = false;

		// ��ȡCPU������
		updateVectorByMatrixGold(_vertexesStatic.pVertex, _vertexesDynamic.nSize, &_joints, _vertexesDynamic.pVertex);

		// ��ȡGPU������
		Vector4 *pVertex = new Vector4[_vertexesDynamic.nSize];
		cudaMemcpy( pVertex, _vertexesDynamic.pVertexDevice, sizeof(Vector4) * _vertexesDynamic.nSize, cudaMemcpyDeviceToHost );
		
		// �ȽϽ��
		bResult = equalVector( _vertexesDynamic.pVertex , _vertexesDynamic.nSize, pVertex );
		printf("%s\n", bResult?"Right":"Wrong");

		// �������٣����ꡢ����
		unInitialize();

		// �鿴ʱ��Ч��
		printf("%d: F=%d, T=%.2f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
	}
	
	// ���������������꣬���յ㡢�ߡ������ʽ
	// ...ʡ��

	return 0;
}

// ���ݳ�ʼ�������ꡢ����
void initialize(int problem_size, int joint_size)
{
	_joints.initialize( joint_size );
#if USE_MEMORY_BUY_TIME
	_vertexesStatic.initialize( problem_size, joint_size );
#endif
	_vertexesDynamic.initialize( problem_size, joint_size );
}

// �������٣����ꡢ����
void unInitialize()
{
	_joints.unInitialize();
#if USE_MEMORY_BUY_TIME
	_vertexesStatic.unInitialize();
#endif
	_vertexesDynamic.unInitialize();
}
