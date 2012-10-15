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
int iClass=6; // �����ģ���ֵ��16M/1G�Դ桢32M/2G�Դ�

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
#if USE_CONSTANT
			// �ı����洢��ʽ����ȫ���Դ��Ϊ�����Դ棬���߾��л���
			constantMemoryUpdate( _joints.pMatrix, JOINT_SIZE );
#else
			cudaMemcpy( _joints.pMatrixDevice, _joints.pMatrix, sizeof(Matrix) * JOINT_SIZE, cudaMemcpyHostToDevice );
#endif
			// ִ�����㣺�������任
			updateVectorByMatrix<<<64, 256>>>(_vertexesStatic.pVertexDevice, PROBLEM_SIZE, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice);
			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();

		// ��֤GPU�������ȷ�ԣ��Ƿ��CPU������һ��
		bool bResult = false;

		// ��ȡCPU������
		updateVectorByMatrixGold(_vertexesStatic.pVertex, PROBLEM_SIZE, _joints.pMatrix, _vertexesDynamic.pVertex);

		// ��ȡGPU������
		Vector4 *pVertex = new Vector4[PROBLEM_SIZE];
		cudaMemcpy( pVertex, _vertexesDynamic.pVertexDevice, sizeof(Vector4) * PROBLEM_SIZE, cudaMemcpyDeviceToHost );
		
		// �ȽϽ��
		bResult = equalVector( _vertexesDynamic.pVertex , PROBLEM_SIZE, pVertex );
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
	_joints.initialize( JOINT_SIZE );
	_vertexesStatic.initialize( PROBLEM_SIZE, JOINT_SIZE );
	_vertexesDynamic.initialize( PROBLEM_SIZE, JOINT_SIZE );
}

// �������٣����ꡢ����
void unInitialize()
{
	_joints.unInitialize();
	_vertexesStatic.unInitialize();
	_vertexesDynamic.unInitialize();
}
