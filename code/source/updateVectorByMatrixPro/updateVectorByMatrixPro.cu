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

// �����в���
int iProblem=6;			// �����ģ���ֵ��16M/512M�Դ桢32M/1G�Դ�
int bAligned = 0;	// �ṹ���Ƿ����
Matrix_Separate_Mode	eSeparate = NO_SEPARATE;	// �ṹ����ģʽ������֡����֡�ȫ���

int bQuiet = 0;		// ��Ĭ��ʽ��������ʾ��Ϣ�������ֻ���ʱ�䣬��λ�Ǻ���
Matrix_Sort_Mode eSort=NO_SORT;			// �����Ծ���idΪ����������ʽ��������˳�����򡢽�������

// ���ݳ�ʼ�������ꡢ����
template<typename T>
void initialize(int problem_size, int joint_size, Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic );

// �������٣����ꡢ����
template<typename T>
void unInitialize( Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  );

// ��ѯÿ��SM����GPU����SP�ĸ���
int _ConvertSMVer2Cores(int major, int minor);

// Ӳ��ӵ�����ĸ����������GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS);

// �����в���˵��
void printHelp(void);

// ִ��ʵ�����
template<typename T>
void runTest(  Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  );

// ����cuda
template<typename T>
void runCuda(  );

// ��֤����Ƿ���ȷ
template<typename T>
bool confirmResult(  Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  );

// ���������в���
bool parseCommand(int argc, const char** argv);

int _tmain(int argc, char** pArgv)
{
	// �����в��������������ο�printHelp
 	const char** argv = (const char**)pArgv;
   shrSetLogFileName ("updateVectorByMatrixPro.txt"); // ������־�ļ�

	// ���������в���
	if( !parseCommand( argc, argv ) )
		return 0;

	if (bAligned)
	{
		// ���ݶ���
		Vertexes<float4>  _vertexesStatic;//��̬��������
		Vertexes<float4>  _vertexesDynamic;//��̬��������
		Joints<float4>		_joints;//�ؽھ���

		runTest<float4>( _joints, _vertexesStatic, _vertexesDynamic );
	}
	else
	{
		// ���ݶ���		
		Vertexes<Vector4>  _vertexesStatic;//��̬��������
		Vertexes<Vector4>  _vertexesDynamic;//��̬��������
		Joints<Vector4>		_joints;//�ؽھ���

		runTest<Vector4>( _joints, _vertexesStatic, _vertexesDynamic );		
	}

	
	// ���������������꣬���յ㡢�ߡ������ʽ
	// ...ʡ��

	return 0;
}

// ���ݳ�ʼ�������ꡢ����
template<typename T>
void initialize(int problem_size, int joint_size, Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic )
{
	joints.initialize( joint_size , eSeparate);
#if USE_MEMORY_BUY_TIME
	vertexesStatic.initialize( problem_size, joint_size, eSort );
#else
	vertexesStatic.initialize( problem_size, joint_size, eSort , false);
#endif
	vertexesDynamic.initialize( problem_size, joint_size, eSort );

	// cuda��ʼ����������ʼ��������Ĭ��ֵ
	int i; // �ж��GPUʱѡ��һ��
	float fGFLOPS = 0.0f;
	i = gpuGetMaxGflopsDeviceId( fGFLOPS );
    if( !bQuiet ) {
		printf("�����������㹫ʽ=sp���� * shaderƵ�� \n\
			�����������Թ���: %0.2f GFLOPS\n", fGFLOPS);
	}
    cudaSetDevice(i);

}

// �������٣����ꡢ����
template<typename T>
void unInitialize( Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic )
{
	joints.unInitialize();

	vertexesStatic.unInitialize();

	vertexesDynamic.unInitialize();
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
			if( !bQuiet ) {
				printf("sp������%d=%d(SM����)*%d(ÿ��SM����SP����), shaderƵ��: %d \n", deviceProp.multiProcessorCount * sm_per_multiproc, 
				deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate);
			}
			SIZE_BLOCK = deviceProp.multiProcessorCount * 2; // �޸Ŀ���Ĭ�����ã���̬����ΪSM��������ȷ��ÿ��SM��2��block
		}
		++current_device;
	}
	fGFLOPS = max_compute_perf * 1.0e-6;
	return max_perf_device;
}
// �����в���˵��
void printHelp(void)
{
    shrLogEx( LOGBOTH|APPENDMODE, 0, "�÷�:  updateVectorByMatrix [ѡ��]...\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "�������任\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "���磺��GPU��ʽִ�о���任�����ݽṹ���ö��뷽ʽ��һ���̴߳���һ������\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "updateVectorByMatrixPro.exe --aligned=1 --multiple=0 \n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "ѡ��:\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "--help\t��ʾ�����˵�\n");

    shrLogEx( LOGBOTH|APPENDMODE, 0, "--quiet=[i]\t��Ĭ��ʽ��������ʾ��Ϣ�������ֻ���ʱ�䣬��λ�Ǻ���\n");   
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1 \n ����Ĭ����Ĭ\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--problem=[i]\t�����ģ����\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2,...,6 \n ���������ģ��7�����Σ�0.25, 0.5, 1, 2, 4, 8, 16, 32��ÿһ����һ������λ�ǰ���\n");

    shrLogEx( LOGBOTH|APPENDMODE, 0, "--aligned=[i]\t����\n");   
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1 \n �����룬����\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--separate=[i]\t�ṹ����ģʽ\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n ����֣����֣�ȫ���\n");
	
	shrLogEx( LOGBOTH|APPENDMODE, 0, "--sort=[i]\t�����Ծ���idΪ����������ʽ\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n ������˳�����򣬽�������\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--multiple=[i]\t�����߳̽���������Ԫ��\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n ����������������������\n");

}

// ����cuda
template<typename T>
void runCuda(  Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  )
{
	globalMemoryUpdate<T>( &joints );

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
	updateVectorByMatrix<T><<<nBlocksPerGrid, nThreadsPerBlock>>>
			( vertexesStatic.pVertexDevice, vertexesDynamic.nSize, joints.pMatrixDevice, vertexesDynamic.pVertexDevice ,
			joints.pMatrixDevicePrevious, eSeparate);

}

// ��֤����Ƿ���ȷ
template<typename T>
bool confirmResult(  Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  )
{
	// ��֤GPU�������ȷ�ԣ��Ƿ��CPU������һ��
	bool bResult = false;

	// ��ȡCPU������
#if SEPERATE_STRUCT_FULLY
	updateVectorByMatrixGoldFully(_vertexesStatic.pVertex, _vertexesDynamic.pVertex, _vertexesDynamic.nSize, _joints.pMatrix, _joints.pMatrixPrevious );
#else
	updateVectorByMatrixGold<T>( vertexesStatic.pVertex, vertexesDynamic.nSize, &joints, vertexesDynamic.pVertex, eSeparate);
#endif
	// ��ȡGPU������
	T *pVertex = new T[vertexesDynamic.nSize];
	cudaMemcpy( pVertex, vertexesDynamic.pVertexDevice, sizeof(T) * vertexesDynamic.nSize, cudaMemcpyDeviceToHost );

	// �ȽϽ��
	bResult = equalVector( vertexesDynamic.pVertex , vertexesDynamic.nSize, pVertex );

	return bResult;
}

// ִ��ʵ�����
template<typename T>
void runTest(  Joints<T>& joints, Vertexes<T>&vertexesStatic, Vertexes<T>&vertexesDynamic  )
{
		StopWatchWin timer;
		int nRepeatPerSecond = 0;// ÿ���ظ���������ʾʱ��Ч��
		
		// �����ģ���Σ�7����64K��256M��4������
		PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iProblem] ;

		// ���ݳ�ʼ�������ꡢ����
		initialize<T>(PROBLEM_SIZE, JOINT_SIZE, joints, vertexesStatic, vertexesDynamic);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			// ִ��ʵ�����
			runCuda<T>( joints, vertexesStatic, vertexesDynamic );

			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}
		timer.stop();
		timer.reset();
		
		// �鿴����Ƿ���ȷ
		bool bResult = confirmResult<T>( joints, vertexesStatic, vertexesDynamic );
		if( !bQuiet ) {
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%s\n", bResult?"Right":"Wrong");
		}
		
		// �������٣����ꡢ����
		unInitialize<T>( joints, vertexesStatic, vertexesDynamic  );

		// �鿴ʱ��Ч��
		if( !bQuiet ) {
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%d: F=%d, T=%.2f ms\n", iProblem+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
		}
		else
		{
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%.2f\n", 10000.0f/nRepeatPerSecond);		
		}
}

// ���������в���
bool parseCommand(int argc, const char** argv)
{
	if(shrCheckCmdLineFlag( argc, argv, "help"))
    {
        printHelp();
		system( "pause" );
        return false;
    }
		
	// ���������в�������ȡ�����ģ --quiet=0
	if(shrCheckCmdLineFlag( argc, argv, "quiet"))
    {
		shrGetCmdLineArgumenti(argc, argv, "quiet", &bQuiet);
	}

	// ���������в�������ȡ�����ģ --class=6
	if(shrCheckCmdLineFlag( argc, argv, "problem"))
    {
		shrGetCmdLineArgumenti(argc, argv, "problem", &iProblem);
	}
	
	// ���������в�������ȡ������ --aligned=0
	if(shrCheckCmdLineFlag( argc, argv, "aligned"))
	{
		shrGetCmdLineArgumenti(argc, argv, "aligned", &bAligned);
	}

	// ���������в���������ṹ����ģʽ --separate=0
    if(shrCheckCmdLineFlag( argc, argv, "separate"))
    {
		int mode;
		shrGetCmdLineArgumenti(argc, argv, "separate", &mode);
		eSeparate = (Matrix_Separate_Mode)mode;
	}
	
	// ���������в���������ṹ����ģʽ --sort=0
    if(shrCheckCmdLineFlag( argc, argv, "sort"))
    {
		int mode;
		shrGetCmdLineArgumenti(argc, argv, "sort", &mode);
		eSort = (Matrix_Sort_Mode)mode;
	}

	if( !bQuiet ) {
		shrLogEx( LOGBOTH|APPENDMODE, 0, "\nOptions begin(���ÿ�ʼ):\n");
		shrLogEx( LOGBOTH|APPENDMODE, 0, "problem=%d(�����ģ��7������)\n", iProblem+1);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "aligned=%d(�����룬����)\n", bAligned);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "separate=%d(����֣����֣�ȫ���)\n", eSeparate);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "sort=%d(������˳�����򣬽�������)\n", eSort);
		
		shrLogEx( LOGBOTH|APPENDMODE, 0, "Options end(���ý���):\n\n");
	}

	return true;
}