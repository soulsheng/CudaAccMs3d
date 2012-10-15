// updateVectorByMatrixPro.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"


#include "Joint.h"
#include "../common/stopwatch_win.h"

// ���ݶ���
Joints		_joints;//�ؽھ���

// ���ݳ�ʼ�������ꡢ����
void initialize(int joint_size);

// �������٣����ꡢ����
void unInitialize();

int _tmain(int argc, _TCHAR* argv[])
{
	
	int nRepeatPerSecond = 0;// ÿ���ظ���������ʾʱ��Ч��
	
	StopWatchWin timer;
	
	{		
		// ���ݳ�ʼ�������ꡢ����
		initialize(JOINT_SIZE*BASE_SIZE);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			cudaMemcpy( _joints.pMatrixDevice, _joints.pMatrix, sizeof(Matrix) * JOINT_SIZE*BASE_SIZE, cudaMemcpyHostToDevice );
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();

		// �������٣����ꡢ����
		unInitialize();

		// �鿴ʱ��Ч��
		printf("%d: F=%d, T=%.1f us\n", SCALE_CLASS, nRepeatPerSecond/10, 10000000.0f/nRepeatPerSecond);
	}
	
	// ���������������꣬���յ㡢�ߡ������ʽ
	// ...ʡ��

	return 0;
}

// ���ݳ�ʼ�������ꡢ����
void initialize(int joint_size)
{
	_joints.initialize( joint_size );
}

// �������٣����ꡢ����
void unInitialize()
{
	_joints.unInitialize();
}
