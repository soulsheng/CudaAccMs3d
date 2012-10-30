// updateVectorByMatrixPro.cpp : 定义控制台应用程序的入口点。
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

int		SIZE_BLOCK=32; // 默认设置，动态设置为SM的两倍，确保每个SM有2个block

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=6; // 问题规模最大值，16M/512M显存、32M/1G显存

bool ALIGNED_STRUCT = false;

// 数据定义
Vertexes  _vertexesStatic;//静态顶点坐标
Vertexes  _vertexesDynamic;//动态顶点坐标
Joints<float4>		_joints1;//关节矩阵
Joints<Vector4>		_joints2;//关节矩阵

// 数据初始化：坐标、矩阵
template<typename T>
void initialize(int problem_size, int joint_size, Joints<T>& joints);

// 数据销毁：坐标、矩阵
template<typename T>
void unInitialize( Joints<T>& joints );

// 查询每个SM包含GPU核心SP的个数
int _ConvertSMVer2Cores(int major, int minor);

// 硬件拥有最大的浮点计算能力GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS);

// 命令行参数说明
void printHelp(void);

// 执行实验过程
template<typename T>
void runTest();

// 验证结果是否正确
template<typename T>
bool confirmResult();

int _tmain(int argc, char** pArgv)
{
	// 命令行参数解析，参数参考printHelp
 	const char** argv = (const char**)pArgv;
   shrSetLogFileName ("updateVectorByMatrixPro.txt"); // 配置日志文件

	if(shrCheckCmdLineFlag( argc, argv, "help"))
    {
        printHelp();
        return 0;
    }

    shrGetCmdLineArgumenti(argc, argv, "class", &iClass);
	
	if(shrCheckCmdLineFlag( argc, argv, "aligned"))
	{
		ALIGNED_STRUCT = true;
	}

	int nRepeatPerSecond = 0;// 每秒重复次数，表示时间效率
	
	StopWatchWin timer;
	
	// 问题规模档次，7档，64K至256M，4倍递增
	PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;
		
	if (ALIGNED_STRUCT)
	{
		// 数据初始化：坐标、矩阵
		initialize<float4>(PROBLEM_SIZE, JOINT_SIZE,_joints1);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			// 执行实验过程
			runTest<float4>();

			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}
		timer.stop();
		timer.reset();
		unInitialize<float4>(_joints1);
		bool bResult = confirmResult<float4>();
		shrLogEx( LOGBOTH|APPENDMODE, 0, "%s\n", bResult?"Right":"Wrong");
	}
	else
	{
		// 数据初始化：坐标、矩阵
		initialize<Vector4>(PROBLEM_SIZE, JOINT_SIZE,_joints2);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			// 执行实验过程
			runTest<Vector4>();

			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}
		timer.stop();
		timer.reset();
		
		// 数据销毁：坐标、矩阵
		unInitialize<Vector4>(_joints2);
		
		// 查看结果是否正确
		bool bResult = confirmResult<Vector4>();
		shrLogEx( LOGBOTH|APPENDMODE, 0, "%s\n", bResult?"Right":"Wrong");
	}

	
	// 查看时间效率
	shrLogEx( LOGBOTH|APPENDMODE, 0, "%d: F=%d, T=%.2f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);

	
	// 输出结果：绘制坐标，按照点、线、面的形式
	// ...省略

	return 0;
}

// 数据初始化：坐标、矩阵
template<typename T>
void initialize(int problem_size, int joint_size, Joints<T>& joints)
{
	joints.initialize( joint_size );
#if USE_MEMORY_BUY_TIME
	_vertexesStatic.initialize( problem_size, joint_size );
#else
	_vertexesStatic.initialize( problem_size, joint_size , false);
#endif
	_vertexesDynamic.initialize( problem_size, joint_size );

	// cuda初始化，若不初始化将采用默认值
	int i; // 有多个GPU时选择一个
	float fGFLOPS = 0.0f;
	i = gpuGetMaxGflopsDeviceId( fGFLOPS );
    printf("计算能力估算公式=sp核数 * shader频率 \n\
			计算能力粗略估算: %0.2f GFLOPS\n", fGFLOPS);
    cudaSetDevice(i);

}

// 数据销毁：坐标、矩阵
template<typename T>
void unInitialize(Joints<T>& joints)
{
	joints.unInitialize();

	_vertexesStatic.unInitialize();

	_vertexesDynamic.unInitialize();
}

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
// 查询每个SM包含GPU核心SP的个数
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
// 硬件拥有最大的浮点计算能力GFLOPS
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
		// clockRate指shader的频率，单位是kHz，即"Clock frequency in kilohertz "，参考：http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/structcudaDeviceProp_dee14230e417cb3059d697d6804da414.html#dee14230e417cb3059d697d6804da414

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
			printf("sp核数：%d=%d(SM个数)*%d(每个SM包含SP个数), shader频率: %d \n", deviceProp.multiProcessorCount * sm_per_multiproc, 
				deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate);
			SIZE_BLOCK = deviceProp.multiProcessorCount * 2; // 修改块数默认设置，动态设置为SM的两倍，确保每个SM有2个block
		}
		++current_device;
	}
	fGFLOPS = max_compute_perf * 1.0e-6;
	return max_perf_device;
}
// 命令行参数说明
void printHelp(void)
{
    shrLog("用法:  updateVectorByMatrix [选项]...\n");
    shrLog("坐标矩阵变换\n");
    shrLog("\n");
    shrLog("例如：用GPU方式执行矩阵变换，数据结构采用对齐方式，一个线程处理一个顶点\n");
    shrLog("updateVectorByMatrixPro.exe --class=6 --aligned --single --buy \n");

    shrLog("\n");
    shrLog("选项:\n");
    shrLog("--help\t显示帮助菜单\n");

	shrLog("  0,1 - 1表示是，0表示否\n");

    shrLog("--aligned\t对齐\n");   
    shrLog("--single\t同一线程处理一个数据元素\n");
    shrLog("--buy\t以空间换时间\n");

	shrLog("--class=[i]\t问题规模档次\n");
    shrLog("  i=0,1,2,...,6 - 代表问题元素的7个档次，0.25, 0.5, 1, 2, 4, 8, 16, 32，每一档翻一倍，单位是百万\n");
}

// 执行实验过程
template<typename T>
void runTest()
{
	globalMemoryUpdate<T>( &_joints1 );

#if !USE_MEMORY_BUY_TIME && _DEBUG
	// 为了确保重复试验得到相同结果，恢复缺省值
	_vertexesDynamic.copy( _vertexesStatic );
#endif

	dim3 nBlocksPerGrid( SIZE_BLOCK ); // 块的数目
	dim3 nThreadsPerBlock( SIZE_THREAD ); // 单块包含线程的数目

#if USE_ELEMENT_SINGLE
	nBlocksPerGrid.y = (PROBLEM_SIZE+nThreadsPerBlock.x - 1)/(nThreadsPerBlock.x * nBlocksPerGrid.x);
#endif

	// 执行运算：坐标矩阵变换
#if SEPERATE_STRUCT

#if SEPERATE_STRUCT_FULLY
	updateVectorByMatrixFully<<<nBlocksPerGrid, nThreadsPerBlock>>>( _vertexesStatic.pVertexDevice,_vertexesDynamic.pVertexDevice, _vertexesDynamic.nSize,
		_joints.nSize, _joints.pMatrixDevice, _joints.pMatrixDevicePrevious);
#else // SEPERATE_STRUCT_FULLY
	updateVectorByMatrix<<<nBlocksPerGrid, nThreadsPerBlock>>>
		(_vertexesStatic.pVertexDevice, _vertexesDynamic.nSize, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice , _joints.pMatrixDevicePrevious);
#endif // SEPERATE_STRUCT_FULLY

#else

		updateVectorByMatrix<T><<<nBlocksPerGrid, nThreadsPerBlock>>>
			(_vertexesStatic.pVertexDevice, _vertexesDynamic.nSize, _joints.pMatrixDevice, _vertexesDynamic.pVertexDevice ,
			_joints.pMatrixDevicePrevious);

#endif
}

// 验证结果是否正确
template<typename T>
bool confirmResult()
{
	// 验证GPU运算的正确性，是否和CPU运算结果一致
	bool bResult = false;

	// 获取CPU运算结果
#if SEPERATE_STRUCT_FULLY
	updateVectorByMatrixGoldFully(_vertexesStatic.pVertex, _vertexesDynamic.pVertex, _vertexesDynamic.nSize, _joints.pMatrix, _joints.pMatrixPrevious );
#else
	updateVectorByMatrixGold<T>(_vertexesStatic.pVertex, _vertexesDynamic.nSize, &_joints, _vertexesDynamic.pVertex);
#endif
	// 获取GPU运算结果
	T *pVertex = new T[_vertexesDynamic.nSize];
	cudaMemcpy( pVertex, _vertexesDynamic.pVertexDevice, sizeof(T) * _vertexesDynamic.nSize, cudaMemcpyDeviceToHost );

	// 比较结果
	bResult = equalVector( _vertexesDynamic.pVertex , _vertexesDynamic.nSize, pVertex );

	return bResult;
}