// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
// 
// WARRANTY DISCLAIMER
// 
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#include "stdafx.h"
#include "CL\cl.h"
#include "utils.h"
//#include "Vertex.h"
//#include "Joint.h"
//for perf. counters
#include <Windows.h>
//#include <omp.h>

#include "MatrixMulVector.h"

//we want to use POSIX functions
#pragma warning( push )
#pragma warning( disable : 4996 )

// OpenCL specific
cl_context	g_context = NULL;
cl_command_queue g_cmd_queue = NULL;
cl_program	g_program = NULL;
cl_kernel	g_kernel = NULL;
cl_kernel	g_kernel4 = NULL;
cl_uint     g_min_align = 0;
cl_device_id g_device_ID =0;
cl_event g_perf_event = NULL;



bool g_bGather4 = false;

bool g_bRunOnPG = false;



#define    MEGA_SIZE     (1<<20)  // Mega, or million
#define    JOINT_SIZE    100

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=2;


void Cleanup()
{
    
    if( g_kernel ) {clReleaseKernel( g_kernel );  g_kernel = NULL;}
    if( g_kernel4 ) {clReleaseKernel( g_kernel4 );  g_kernel4 = NULL;}
    if( g_program ) {clReleaseProgram( g_program );  g_program = NULL;}
    if( g_cmd_queue ) {clReleaseCommandQueue( g_cmd_queue );  g_cmd_queue = NULL;}
    if( g_context ) {clReleaseContext( g_context );  g_context = NULL;}
	if( g_perf_event ){clReleaseEvent(g_perf_event);g_perf_event =NULL;}
    //host memory
//    if(g_pfInput) {_aligned_free( g_pfInput ); g_pfInput = NULL;}
//    if(g_pfRegularOutput) {_aligned_free( g_pfRegularOutput ); g_pfRegularOutput = NULL;}
    //if(g_pfOCLOutput) {_aligned_free( g_pfOCLOutput ); g_pfOCLOutput = NULL;}
	//unInitialize();
}

bool Setup_OpenCL( const char *program_source )
{
    cl_device_id devices[16];
    size_t cb;
    cl_uint size_ret = 0;
    cl_int err;
    int num_cores;
    char device_name[128] = {0};

	static const char buildOpts[] = "-cl-fast-relaxed-math";

	if(g_bRunOnPG)
	{
		printf("Trying to run on a Processor Graphics \n");
	}
	else
	{
		printf("Trying to run on a CPU \n");
	}

	cl_platform_id intel_platform_id = GetIntelOCLPlatform();
    if( intel_platform_id == NULL )
    {
        printf("ERROR: Failed to find Intel OpenCL platform.\n");
        return false;
    }

    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

    // create the OpenCL context on a CPU/PG 
	if(g_bRunOnPG)
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	}
	else
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
	}
    if (g_context == (cl_context)0)
        return false;

    // get the list of CPU devices associated with context
    err = clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    clGetContextInfo(g_context, CL_CONTEXT_DEVICES, cb, devices, NULL);
	g_cmd_queue = clCreateCommandQueue(g_context, devices[0], 0, NULL);
    if (g_cmd_queue == (cl_command_queue)0)
    {
        Cleanup();
        return false;
    }

    char *sources = ReadSources(program_source);	//read program .cl source file
    g_program = clCreateProgramWithSource(g_context, 1, (const char**)&sources, NULL, NULL);
    if (g_program == (cl_program)0)
    {
        printf("ERROR: Failed to create Program with source...\n");
        Cleanup();
        free(sources);
        return false;
    }

	err = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to build program...\n");
        BuildFailLog(g_program, devices[0]);
        Cleanup();
        free(sources);
        return false;
    }
#if !VECTOR_FLOAT4
    g_kernel = clCreateKernel(g_program, "updateVectorByMatrix", NULL);
#else
	g_kernel = clCreateKernel(g_program, "updateVectorByMatrix4", NULL);
#endif
    if (g_kernel == (cl_kernel)0)
    {
        printf("ERROR: Failed to create kernel...\n");
        Cleanup();
        free(sources);
        return false;
    }
    g_kernel4 = clCreateKernel(g_program, "SimpleKernel4", NULL);
    if (g_kernel4 == (cl_kernel)0)
    {
        printf("ERROR: Failed to create second kernel...\n");
        Cleanup();
        free(sources);
        return false;
    }
    free(sources);

    // use first device ID
    g_device_ID = devices[0];
    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_NAME, 128, device_name, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (device name)...\n");
        Cleanup();
        return false;
    }
    printf("Using device %s...\n", device_name);

    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (max compute units)...\n");
        Cleanup();
        return false;
    }
    printf("Using %d compute units...\n", num_cores);


    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &g_min_align, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (max memory base address align size)...\n");
        Cleanup();
        return false;
    }
    g_min_align /= 8; //in bytes
    printf("Buffer alignment required for zero-copying is %d bytes (CL_DEVICE_MEM_BASE_ADDR_ALIGN)\n\n", g_min_align);

    return true; // success...
}


void Usage()
{
    printf("Usage: SimpleOptimizations.exe [--h] [-t <TaskSize>][-l <GroupSize>] [-r] [-p] [-a] [-w] [-v] [-g]\n");
    printf("  where, --h prints this message\n");
    printf("    <TaskSize> is task size (also global size)\n");
    printf("    <GroupSize> is work group size (aka local size)\n");
    printf("    -r relaxed math enabled\n");
    printf("    -p host pointers/buffer-mapping enabled\n");
    printf("    -a auto-selected work group size enbaled, [-l] 'local size' option will be ignored in this case\n");
    printf("    -f OpenCL profiling will be enabled\n");
    printf("    -w additional \"warming\" kernel run enabled \n");
    printf("    -v \"gather4\" kernel version\n");
    printf("    -g run on Processor Graphics\n");
    exit(-1);
}

// main execution routine - perform simple math on float vectors
int _tmain(int argc, _TCHAR* argv[])
{
    //parse command line
    int argn = 1;
    while (argn < argc)
    {
       if (_tcscmp(argv[argn], _T("-v")) == 0)
        {
            g_bGather4 = true;
            argn++;
        }
		else if (_tcscmp(argv[argn], _T("-g")) == 0)
		{
			g_bRunOnPG = true;
			argn++;
		}
		else if (_tcscmp(argv[argn], _T("-index")) == 0)
		{
			if(++argn==argc)
				Usage();
			iClass = _ttoi(argv[argn]);
			argn++;
		}
        else
        {
            argn++;
        }
    }
    if( argc < 2 )
    {
        printf("No command line arguments specified, using default values.\n");
    }

    //initialize Open CL objects (context, queue, etc.)
    if( Setup_OpenCL("SimpleOptimizations.cl")!=true )
        return -1;
	
	// 问题规模档次，7档，64K至256M，4倍递增
	PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;

    //	set input array to random legal values

	// 数据初始化：坐标、矩阵
	CMatrixMulVector	mvm;	
	mvm.initialize(PROBLEM_SIZE, JOINT_SIZE);//initialize(PROBLEM_SIZE, JOINT_SIZE);


	cl_kernel	kernel = g_kernel;
	if(g_bGather4)
	{
		kernel = g_kernel4;
	}
    //do simple math
    if( mvm.ExecuteKernel( g_context, g_device_ID, kernel, g_cmd_queue )!=true)
    {
        printf("Failed executing OpenCL kernel...\n");
        Cleanup();
        return -1;
    }


	printf("Executing reference...");
	if(g_bGather4)	
		mvm.ExecuteNativeSSE();
	else
		mvm.ExecuteNativeCPP();

	printf("Done\n\n");


    //Do verification
    printf("Performing verification...\n");
 	bool result = mvm.verifyEqual( );
	printf("%s", !result ?"ERROR: Verification failed.\n":"Verification succeeded.\n");
    
	mvm.unInitialize();
    Cleanup();
   
	return 0;
}

#pragma warning( pop )
