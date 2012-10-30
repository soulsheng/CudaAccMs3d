// updateVectorByMatrix.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "../common/shrUtils.h"
#include "updateVectorByMatrix.h"

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // �����ģ���Σ�8����250K��32M��2������
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// �����ģ, ��ʼ��Ϊ1M����һ����
int iClass=6;

bool USE_OPENMP = false;

// ���ݶ���
Vertexes  _vertexesStatic;//��̬��������
Vertexes  _vertexesDynamic;//��̬��������
Joints		_joints;//�ؽھ���

// ���ݳ�ʼ�������ꡢ����
void initialize(int problem_size, int joint_size);

// �������任
void updateVectorByMatrix(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut);

// �������٣����ꡢ����
void unInitialize();

// �����в���˵��
void printHelp(void);

int _tmain(int argc, char** pArgv)
{
	// �����в��������������ο�printHelp
	const char** argv = (const char**)pArgv;
	shrSetLogFileName ("updateVectorByMatrix.txt"); // ������־�ļ�

	if(shrCheckCmdLineFlag( argc, argv, "help"))
	{
		printHelp();
		return 0;
	}

	shrGetCmdLineArgumenti(argc, argv, "class", &iClass);

	if(shrCheckCmdLineFlag( argc, argv, "openmp"))
	{
		USE_OPENMP = true;
	}

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
			updateVectorByMatrix(_vertexesStatic.pVertex, PROBLEM_SIZE, _joints.pMatrix, _vertexesDynamic.pVertex, USE_OPENMP);
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();
		
		// �������٣����ꡢ����
		unInitialize();

		// �鿴ʱ��Ч��
		shrLogEx( LOGBOTH|APPENDMODE, 0, "%d: F=%d, T=%.2f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
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

// �����в���˵��
void printHelp(void)
{
	shrLog("�÷�:  updateVectorByMatrix [ѡ��]...\n");
	shrLog("�������任\n");
	shrLog("\n");
	shrLog("���磺��CPU��ʽִ�о���任�������ģ�ǵ�7����1ǧ6���򣩣�����OpenMP���̣߳��Կռ任ʱ��\n");
	shrLog("updateVectorByMatrix.exe --class=6 --openmp --buy \n");

	shrLog("\n");
	shrLog("ѡ��:\n");
	shrLog("--help\t��ʾ�����˵�\n");

	shrLog("--openmp\t���û���OpenMP�Ķ��߳�\n");  
	shrLog("--buy\t�Կռ任ʱ��\n");

	shrLog("--class=[i]\t�����ģ����\n");
	shrLog("  i=0,1,2,...,6 - ��������Ԫ�ص�7�����Σ�0.25, 0.5, 1, 2, 4, 8, 16, ÿһ����һ������λ�ǰ���\n");
}
