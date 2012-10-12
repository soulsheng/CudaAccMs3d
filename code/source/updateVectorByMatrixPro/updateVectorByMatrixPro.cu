// updateVectorByMatrixPro.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // �����ģ���Σ�8����250K��32M��2������
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// �����ģ, ��ʼ��Ϊ1M����һ����
int iClass=6; // �����ģ���ֵ��16M/1G�Դ桢32M/2G�Դ�

// ���ݶ���
Vertexes  _vertexesStatic;//��̬��������
Vertexes  _vertexesDynamic;//��̬��������
Joints		_joints;//�ؽھ���

// ���ݳ�ʼ�������ꡢ����
void initialize(int problem_size, int joint_size);

// �������任
__global__ void updateVectorByMatrix(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut);

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
			// ִ�����㣺�������任
			updateVectorByMatrix<<<64, 256>>>(_vertexesStatic.pVertexDevice, PROBLEM_SIZE, _joints.pMatrix, _vertexesDynamic.pVertexDevice);
			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();
		
		// �������٣����ꡢ����
		unInitialize();

		// �鿴ʱ��Ч��
		printf("%d: F=%d, T=%.1f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
	}
	
	// ���������������꣬���յ㡢�ߡ������ʽ
	// ...ʡ��

	return 0;
}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
__global__ void updateVectorByMatrix(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut){
	const int indexBase = blockIdx.x * blockDim.x + threadIdx.x;
	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
		float4   vertexIn, vertexOut;
		float3   matrix[3];
		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
		vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int
		matrix[0] = pMatrix[matrixIndex][0];
		matrix[1] = pMatrix[matrixIndex][1];
		matrix[2] = pMatrix[matrixIndex][2];

		// ִ�в�����������ִ�о���任���õ�������
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z ; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z ; 

		// д����������������
		pVertexOut[i] = vertexOut;
	}
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
