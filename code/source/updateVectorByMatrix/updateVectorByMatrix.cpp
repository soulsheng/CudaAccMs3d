// updateVectorByMatrix.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include "../common/stopwatch_win.h"
#include "../common/shrUtils.h"
#include "updateVectorByMatrix.h"

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=6;

bool USE_OPENMP = false;

// 数据定义
Vertexes  _vertexesStatic;//静态顶点坐标
Vertexes  _vertexesDynamic;//动态顶点坐标
Joints		_joints;//关节矩阵

// 数据初始化：坐标、矩阵
void initialize(int problem_size, int joint_size);

// 坐标矩阵变换
void updateVectorByMatrix(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut);

// 数据销毁：坐标、矩阵
void unInitialize();

// 命令行参数说明
void printHelp(void);

int _tmain(int argc, char** pArgv)
{
	// 命令行参数解析，参数参考printHelp
	const char** argv = (const char**)pArgv;
	shrSetLogFileName ("updateVectorByMatrix.txt"); // 配置日志文件

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
			// 执行运算：坐标矩阵变换
			updateVectorByMatrix(_vertexesStatic.pVertex, PROBLEM_SIZE, _joints.pMatrix, _vertexesDynamic.pVertex, USE_OPENMP);
			nRepeatPerSecond ++;
		}

		timer.stop();
		timer.reset();
		
		// 数据销毁：坐标、矩阵
		unInitialize();

		// 查看时间效率
		shrLogEx( LOGBOTH|APPENDMODE, 0, "%d: F=%d, T=%.2f ms\n", iClass+1, nRepeatPerSecond/10, 10000.0f/nRepeatPerSecond);
	}
	
	// 输出结果：绘制坐标，按照点、线、面的形式
	// ...省略

	return 0;
}

// 数据初始化：坐标、矩阵
void initialize(int problem_size, int joint_size)
{
	_joints.initialize( JOINT_SIZE );
	_vertexesStatic.initialize( PROBLEM_SIZE, JOINT_SIZE );
	_vertexesDynamic.initialize( PROBLEM_SIZE, JOINT_SIZE );
}

// 数据销毁：坐标、矩阵
void unInitialize()
{
	_joints.unInitialize();
	_vertexesStatic.unInitialize();
	_vertexesDynamic.unInitialize();
}

// 命令行参数说明
void printHelp(void)
{
	shrLog("用法:  updateVectorByMatrix [选项]...\n");
	shrLog("坐标矩阵变换\n");
	shrLog("\n");
	shrLog("例如：用CPU方式执行矩阵变换，问题规模是第7档（1千6百万），采用OpenMP多线程，以空间换时间\n");
	shrLog("updateVectorByMatrix.exe --class=6 --openmp --buy \n");

	shrLog("\n");
	shrLog("选项:\n");
	shrLog("--help\t显示帮助菜单\n");

	shrLog("--openmp\t采用基于OpenMP的多线程\n");  
	shrLog("--buy\t以空间换时间\n");

	shrLog("--class=[i]\t问题规模档次\n");
	shrLog("  i=0,1,2,...,6 - 代表问题元素的7个档次，0.25, 0.5, 1, 2, 4, 8, 16, 每一档翻一倍，单位是百万\n");
}
