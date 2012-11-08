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
#define     SIZE_SHARE_MEMORY_DYNAMIC	0  // 10K（gts250）  40K（gtx670），为了稳定sp使用率为1/3，即每个sm只运行1个block
int		SIZE_BLOCK=32; // 默认设置，动态设置为SM的两倍，确保每个SM有2个block

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万

// 命令行参数
int iProblem=3;			// 问题规模最大值，16M/512M显存、32M/1G显存
int bAligned = 1;	// 结构体是否对齐
Matrix_Separate_Mode	eSeparate = HALF_SEPARATE;	// 结构体拆分模式，不拆分、半拆分、全拆分

int bQuiet = 1;		// 静默方式，屏蔽提示信息的输出，只输出时间，单位是毫秒
Matrix_Sort_Mode eSort=SERIAL_SORT;			// 顶点以矩阵id为索引的排序方式，不排序、顺序排序、交叉排序
Matrix_Memory_Mode	eMemory=SHARED_MEMORY;	// 矩阵存储位置，全局显存、常量显存、共享显存

// 数据初始化：坐标、矩阵
template<typename F4, typename F1>
void initialize(int problem_size, int joint_size, Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic );

// 数据销毁：坐标、矩阵
template<typename F4, typename F1>
void unInitialize( Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  );

// 查询每个SM包含GPU核心SP的个数
int _ConvertSMVer2Cores(int major, int minor);

// 硬件拥有最大的浮点计算能力GFLOPS
int gpuGetMaxGflopsDeviceId(float& fGFLOPS);

// 命令行参数说明
void printHelp(void);

// 执行实验过程
template<typename F4, typename F1>
void runTest(  Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  );

// 调用cuda
template<typename F4, typename F1>
void runCuda(  );

// 验证结果是否正确
template<typename F4, typename F1>
bool confirmResult(  Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  );

// 解析命令行参数
bool parseCommand(int argc, const char** argv);

int _tmain(int argc, char** pArgv)
{
	// 命令行参数解析，参数参考printHelp
 	const char** argv = (const char**)pArgv;
   shrSetLogFileName ("updateVectorByMatrixPro.txt"); // 配置日志文件

	// 解析命令行参数
	if( !parseCommand( argc, argv ) )
		return 0;

	if (bAligned)
	{
		// 数据定义
		Vertexes<float4>  _vertexesStatic;//静态顶点坐标
		Vertexes<float4>  _vertexesDynamic;//动态顶点坐标
		Joints<float1>		_joints;//关节矩阵

		runTest<float4, float1>( _joints, _vertexesStatic, _vertexesDynamic );
	}
	else
	{
		// 数据定义		
		Vertexes<Vector4>  _vertexesStatic;//静态顶点坐标
		Vertexes<Vector4>  _vertexesDynamic;//动态顶点坐标
		Joints<Vector1>		_joints;//关节矩阵

		runTest<Vector4, Vector1>( _joints, _vertexesStatic, _vertexesDynamic );		
	}

	
	// 输出结果：绘制坐标，按照点、线、面的形式
	// ...省略

	return 0;
}

// 数据初始化：坐标、矩阵
template<typename F4, typename F1>
void initialize(int problem_size, int joint_size, Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic )
{
	joints.initialize( joint_size , eSeparate);
#if USE_MEMORY_BUY_TIME
	vertexesStatic.initialize( problem_size, joint_size );
	vertexesStatic.setDefault( eSort );
#else
	vertexesStatic.initialize( problem_size, joint_size, false);
	vertexesStatic.setDefault( eSort, false );
#endif
	vertexesDynamic.initialize( problem_size, joint_size );
	vertexesDynamic.copy( vertexesStatic );

	// cuda初始化，若不初始化将采用默认值
	int i; // 有多个GPU时选择一个
	float fGFLOPS = 0.0f;
	i = gpuGetMaxGflopsDeviceId( fGFLOPS );
    if( !bQuiet ) {
		printf("计算能力估算公式=sp核数 * shader频率 \n\
			计算能力粗略估算: %0.2f GFLOPS\n", fGFLOPS);
	}
    cudaSetDevice(i);

}

// 数据销毁：坐标、矩阵
template<typename F4, typename F1>
void unInitialize( Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic )
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

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate * 2;
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
			if( !bQuiet ) {
				printf("sp核数：%d=%d(SM个数)*%d(每个SM包含SP个数), shader频率: %d \n", deviceProp.multiProcessorCount * sm_per_multiproc, 
				deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate);
			}
			SIZE_BLOCK = deviceProp.maxThreadsPerMultiProcessor / SIZE_THREAD * deviceProp.multiProcessorCount * 2; // 修改块数默认设置，动态设置为SM的两倍，确保每个SM有2个block
		}
		++current_device;
	}
	fGFLOPS = max_compute_perf * 1.0e-6;
	return fGFLOPS;
}
// 命令行参数说明
void printHelp(void)
{
    shrLogEx( LOGBOTH|APPENDMODE, 0, "用法:  updateVectorByMatrix [选项]...\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "坐标矩阵变换\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "例如：用GPU方式执行矩阵变换，数据结构采用对齐方式，一个线程处理一个顶点\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "updateVectorByMatrixPro.exe --aligned=1 --multiple=0 \n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "选项:\n");
    shrLogEx( LOGBOTH|APPENDMODE, 0, "--help\t显示帮助菜单\n");

    shrLogEx( LOGBOTH|APPENDMODE, 0, "--quiet=[i]\t静默方式，屏蔽提示信息的输出，只输出时间，单位是毫秒\n");   
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1 \n 不静默，静默\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--problem=[i]\t问题规模档次\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2,...,6 \n 代表问题规模的7个档次，0.25, 0.5, 1, 2, 4, 8, 16, 32，每一档翻一倍，单位是百万\n");

    shrLogEx( LOGBOTH|APPENDMODE, 0, "--aligned=[i]\t对齐\n");   
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1 \n 不对齐，对齐\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--separate=[i]\t结构体拆分模式\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n 不拆分，半拆分，全拆分\n");
	
	shrLogEx( LOGBOTH|APPENDMODE, 0, "--sort=[i]\t顶点以矩阵id为索引的排序方式\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n 不排序，顺序排序，交叉排序\n");
	
	shrLogEx( LOGBOTH|APPENDMODE, 0, "--memory=[i]\t矩阵存储位置\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n 全局显存、常量显存、共享显存\n");

	shrLogEx( LOGBOTH|APPENDMODE, 0, "--multiple=[i]\t单个线程解决多个问题元素\n");
	shrLogEx( LOGBOTH|APPENDMODE, 0, "  i=0,1,2 \n 单个，多个连续，多个交替\n");

}

// 调用cuda
template<typename F4, typename F1>
void runCuda(  Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  )
{
#if KERNEL_MEMORY_PREPARE
	globalMemoryUpdate<F1>( &joints, eSeparate, eMemory, bAligned );
#endif

#if !USE_MEMORY_BUY_TIME && _DEBUG
	// 为了确保重复试验得到相同结果，恢复缺省值
	_vertexesDynamic.copy( _vertexesStatic );
#endif

	dim3 nBlocksPerGrid( SIZE_BLOCK ); // 块的数目
	dim3 nThreadsPerBlock( SIZE_THREAD ); // 单块包含线程的数目

#if USE_ELEMENT_SINGLE
	nBlocksPerGrid.y = (PROBLEM_SIZE+nThreadsPerBlock.x - 1)/(nThreadsPerBlock.x * nBlocksPerGrid.x);
#endif

	
	int nSizeSharedMemoryDynamic = (1<<10) * SIZE_SHARE_MEMORY_DYNAMIC;  // 10k 或者 40k
	// 执行运算：坐标矩阵变换
	switch( eMemory )
	{
		case SHARED_MEMORY: 
			{
				updateVectorByMatrixShared<F4><<<nBlocksPerGrid, nThreadsPerBlock, nSizeSharedMemoryDynamic>>>
					( vertexesStatic.pVertexDevice, vertexesDynamic.nSize, (F4*)joints.pMatrixDevice, vertexesDynamic.pVertexDevice ,
					(F4*)joints.pMatrixDevicePrevious, eSeparate );
			}
			break;

		case CONSTANT_MEMORY: 
			{
				updateVectorByMatrixConst<F4><<<nBlocksPerGrid, nThreadsPerBlock>>>
					( vertexesStatic.pVertexDevice, vertexesDynamic.nSize, vertexesDynamic.pVertexDevice , eSeparate, bAligned );
			}
			break;

		default:
			{
				updateVectorByMatrix<F4, F1><<<nBlocksPerGrid, nThreadsPerBlock, nSizeSharedMemoryDynamic>>>
					( vertexesStatic.pVertexDevice, vertexesDynamic.nSize, joints.pMatrixDevice, vertexesDynamic.pVertexDevice ,
					joints.pMatrixDevicePrevious, eSeparate);
			}
			break;
	}//switch

}

// 验证结果是否正确
template<typename F4, typename F1>
bool confirmResult(  Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  )
{
	// 验证GPU运算的正确性，是否和CPU运算结果一致
	bool bResult = false;

	// 获取CPU运算结果
#if SEPERATE_STRUCT_FULLY
	updateVectorByMatrixGoldFully(_vertexesStatic.pVertex, _vertexesDynamic.pVertex, _vertexesDynamic.nSize, _joints.pMatrix, _joints.pMatrixPrevious );
#else
	updateVectorByMatrixGold<F4>( vertexesStatic.pVertex, vertexesDynamic.nSize, &joints, vertexesDynamic.pVertex, eSeparate);
#endif
	// 获取GPU运算结果
	F4 *pVertex = new F4[vertexesDynamic.nSize];
	cudaMemcpy( pVertex, vertexesDynamic.pVertexDevice, sizeof(F4) * vertexesDynamic.nSize, cudaMemcpyDeviceToHost );

	// 比较结果
	bResult = equalVector( vertexesDynamic.pVertex , vertexesDynamic.nSize, pVertex );

	return bResult;
}

// 执行实验过程
template<typename F4, typename F1>
void runTest(  Joints<F1>& joints, Vertexes<F4>&vertexesStatic, Vertexes<F4>&vertexesDynamic  )
{
		StopWatchWin timer;
		int nRepeatPerSecond = 0;// 每秒重复次数，表示时间效率
		
		// 问题规模档次，7档，64K至256M，4倍递增
		PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iProblem] ;

		// 数据初始化：坐标、矩阵
		initialize<F4, F1>(PROBLEM_SIZE, JOINT_SIZE, joints, vertexesStatic, vertexesDynamic);
		timer.start();

		while ( timer.getTime() < 10000  )
		{
			// 执行实验过程
			runCuda<F4, F1>( joints, vertexesStatic, vertexesDynamic );

			cudaDeviceSynchronize();
			nRepeatPerSecond ++;
		}
		timer.stop();
		timer.reset();
		
		// 查看结果是否正确
		bool bResult = confirmResult<F4, F1>( joints, vertexesStatic, vertexesDynamic );
		if( !bQuiet ) {
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%s\n", bResult?"Right":"Wrong");
		}
		
		// 数据销毁：坐标、矩阵
		unInitialize<F4, F1>( joints, vertexesStatic, vertexesDynamic  );

		// 查看时间效率
		if( !bQuiet ) {
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%d: F=%d, T=%.2f ms\n", iProblem+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
		}
		else
		{
			shrLogEx( LOGBOTH|APPENDMODE, 0, "%.2f\n", 10000.0f/nRepeatPerSecond);		
		}
}

// 解析命令行参数
bool parseCommand(int argc, const char** argv)
{
	if(shrCheckCmdLineFlag( argc, argv, "help"))
    {
        printHelp();
		system( "pause" );
        return false;
    }
		
	// 解析命令行参数，获取问题规模 --quiet=0
	if(shrCheckCmdLineFlag( argc, argv, "quiet"))
    {
		shrGetCmdLineArgumenti(argc, argv, "quiet", &bQuiet);
	}

	// 解析命令行参数，获取问题规模 --class=6
	if(shrCheckCmdLineFlag( argc, argv, "problem"))
    {
		shrGetCmdLineArgumenti(argc, argv, "problem", &iProblem);
	}
	
	// 解析命令行参数，获取对齐标记 --aligned=0
	if(shrCheckCmdLineFlag( argc, argv, "aligned"))
	{
		shrGetCmdLineArgumenti(argc, argv, "aligned", &bAligned);
	}

	// 解析命令行参数，矩阵结构体拆分模式 --separate=0
    if(shrCheckCmdLineFlag( argc, argv, "separate"))
    {
		int mode;
		shrGetCmdLineArgumenti(argc, argv, "separate", &mode);
		eSeparate = (Matrix_Separate_Mode)mode;
	}
	
	// 解析命令行参数，矩阵结构体拆分模式 --sort=0
    if(shrCheckCmdLineFlag( argc, argv, "sort"))
    {
		int mode;
		shrGetCmdLineArgumenti(argc, argv, "sort", &mode);
		eSort = (Matrix_Sort_Mode)mode;
	}
	
	// 矩阵存储位置 --memory=0
    if(shrCheckCmdLineFlag( argc, argv, "memory"))
    {
		int mode;
		shrGetCmdLineArgumenti(argc, argv, "memory", &mode);
		eMemory = (Matrix_Memory_Mode)mode;
	}

	if( !bQuiet ) {
		shrLogEx( LOGBOTH|APPENDMODE, 0, "\nOptions begin(配置开始):\n");
		shrLogEx( LOGBOTH|APPENDMODE, 0, "problem=%d(问题规模的7个档次)\n", iProblem+1);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "aligned=%d(不对齐，对齐)\n", bAligned);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "separate=%d(不拆分，半拆分，全拆分)\n", eSeparate);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "sort=%d(不排序，顺序排序，交叉排序)\n", eSort);
		shrLogEx( LOGBOTH|APPENDMODE, 0, "memory=%d(全局显存、常量显存、共享显存)\n", eMemory);
		
		shrLogEx( LOGBOTH|APPENDMODE, 0, "Options end(配置结束):\n\n");
	}

	return true;
}