// updateVectorByMatrixPro.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "../common/shrUtils.h"
#include "updateVectorByMatrixPro.cuh"
#include "updateVectorByMatrix.h"

#define		SIZE_THREAD	256

int		SIZE_BLOCK=32; // Ĭ�����ã���̬����ΪSM��������ȷ��ÿ��SM��2��block

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

// ��ѯÿ��SM����GPU����SP�ĸ���
int _ConvertSMVer2Cores(int major, int minor);

// Ӳ��ӵ�����ĸ����������GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS);


int _tmain(int argc, _TCHAR* argv[])
{
	// �����в�������


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
			
#if !USE_MEMORY_BUY_TIME && _DEBUG
			// Ϊ��ȷ���ظ�����õ���ͬ������ָ�ȱʡֵ
			_vertexesDynamic.copy( _vertexesStatic );
#endif

			dim3 nBlocksPerGrid( SIZE_BLOCK ); // �����Ŀ
			dim3 nThreadsPerBlock( SIZE_THREAD ); // ��������̵߳���Ŀ

#if USE_ELEMENT_SINGLE
			nBlocksPerGrid.y = (PROBLEM_SIZE+nThreadsPerBlock.x - 1)/(nThreadsPerBlock.x * nBlocksPerGrid.x);
#endif

			// ִ�����㣺�������任
#if SEPERATE_STRUCT

#if SEPERATE_STRUCT_FULLY
			updateVectorByMatrixFully<<<nBlocksPerGrid, nThreadsPerBlock>>>( _vertexesStatic.pVertexDevice,_vertexesDynamic.pVertexDevice, _vertexesDynamic.nSize,
				_joints.nSize, _joints.pMatrixDevice, _joints.pMatrixDevicePrevious);
#else // SEPERATE_STRUCT_FULLY
			updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock>>>
				(_vertexesStatic.pVertexDevice, _vertexesDynamic.nSize, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice , _joints.pMatrixDevicePrevious);
#endif // SEPERATE_STRUCT_FULLY

#else

			updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock>>>
				(_vertexesStatic.pVertexDevice, _vertexesDynamic.nSize, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice ,
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
#if SEPERATE_STRUCT_FULLY
		updateVectorByMatrixGoldFully(_vertexesStatic.pVertex, _vertexesDynamic.pVertex, _vertexesDynamic.nSize, _joints.pMatrix, _joints.pMatrixPrevious );
#else
		updateVectorByMatrixGold(_vertexesStatic.pVertex, _vertexesDynamic.nSize, &_joints, _vertexesDynamic.pVertex);
#endif
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
#else
	_vertexesStatic.initialize( problem_size, joint_size , false);
#endif
	_vertexesDynamic.initialize( problem_size, joint_size );

	// cuda��ʼ����������ʼ��������Ĭ��ֵ
	int i; // �ж��GPUʱѡ��һ��
	float fGFLOPS = 0.0f;
	i = gpuGetMaxGflopsDeviceId( fGFLOPS );
    printf("�����������㹫ʽ=sp���� * shaderƵ�� \n\
			�����������Թ���: %0.2f GFLOPS\n", fGFLOPS);
    cudaSetDevice(i);

}

// �������٣����ꡢ����
void unInitialize()
{
	_joints.unInitialize();

	_vertexesStatic.unInitialize();

	_vertexesDynamic.unInitialize();
}

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
// ��ѯÿ��SM����GPU����SP�ĸ���
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
       int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
       int Cores;
    } sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
       if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
          return nGpuArchCoresPerSM[index].Cores;
       }	
       index++;
    }
    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
    return -1;
}
// Ӳ��ӵ�����ĸ����������GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS)
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );
    
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{
			sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		// clockRateָshader��Ƶ�ʣ���λ��kHz����"Clock frequency in kilohertz "���ο���http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/structcudaDeviceProp_dee14230e417cb3059d697d6804da414.html#dee14230e417cb3059d697d6804da414

		if( compute_perf  > max_compute_perf )
		{
			// If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 )
			{
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch)
				{
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
			else
			{
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
			printf("sp������%d=%d(SM����)*%d(ÿ��SM����SP����), shaderƵ��: %d \n", deviceProp.multiProcessorCount * sm_per_multiproc, 
				deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate);
			SIZE_BLOCK = deviceProp.multiProcessorCount * 2; // �޸Ŀ���Ĭ�����ã���̬����ΪSM��������ȷ��ÿ��SM��2��block
		}
		++current_device;
	}
	fGFLOPS = max_compute_perf * 1.0e-6;
	return max_perf_device;
}
