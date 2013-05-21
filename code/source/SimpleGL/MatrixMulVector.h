#pragma once

#include "CL\cl.h"
#include <GL/glew.h>
#include "Vertex.h"
#include "Joint.h"
#include <SDKCommon.hpp>
#include <map>

#define  SIZE_VBO		3

typedef std::multimap<std::string, double> TimerList;
typedef std::multimap<std::string, double>::iterator TimerListItr;

class CMatrixMulVector
{
public:
	CMatrixMulVector();
	~CMatrixMulVector();

public:
	// ���ݳ�ʼ�������ꡢ����
	void initialize(int sizeVertex, int sizeJoint, streamsdk::SDKCommon * pSampleCommon, TimerList* pTL);
	
	// �������٣����ꡢ����
	void unInitialize();

	// ִ�и����㷨
	void ExecuteNativeCPP();
	void ExecuteNativeSSE();
	void ExecuteNativeCPPOMP();
	void ExecuteNativeSSEOMP();
	
	// ��֤����Ƿ���ȷ
	bool verifyEqual();

	void SetupKernelVBO(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue,int* nLocationAttrib);
	void SetupKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue);
	void SetupWorksize( );
	bool ExecuteKernel();
	bool ExecuteKernelVBO();

	void  renderVBO();

public:
	/**
     * Timer functions
     */
    int createTimer()
    {
        return sampleCommon->createTimer();
    }

    int resetTimer(int handle)
    {
        return sampleCommon->resetTimer(handle);
    }

    int startTimer(int handle)
    {
        return sampleCommon->startTimer(handle);
    }

	int stopTimer(int handle)
	{
		return sampleCommon->stopTimer(handle);
	}

    double readTimer(int handle)
    {
        return sampleCommon->readTimer(handle);
    }
	void insertTimer(std::string, double);

private:
	// ����任
	void MatrixVectorMul(float* vIn, float* vOut, float* mat);
	void MatrixVectorMul(cl_float4* vIn, cl_float4* vOut, cl_float4* mat);

	// ��֤����Ƿ���ȷ
	bool verifyEqual(float *v, float* vRef, int size);
	bool verifyEqual(cl_float4 *v, cl_float4* vRef, int size);

public:
	// ���ݶ���
	Vertexes  _vertexesStatic;//��̬��������
	Vertexes  _vertexesDynamic, _vertexesDynamicRef;//��̬��������
	Joints		_joints;//�ؽھ���

	//	cl_mem objects used as parameters for kernels
	cl_mem g_pfInputBuffer ;
	cl_mem g_pfOCLOutputBuffer ;
	cl_mem g_pfOCLIndex , g_pfOCLWeight;
	cl_mem g_pfOCLMatrix ;

	cl_context	_context ;
	cl_device_id _device_ID ;
	cl_command_queue _cmd_queue ;
	cl_kernel	_kernel;

	size_t globalWorkSize[2];
	size_t localWorkSize[2];

	// vbo
	GLuint vertexVAO;                   /**< Vertex Array object */
	GLuint vertexObj[SIZE_VBO];                   /**< Vertex object */
	
	int*   _locationAttrib;

	// Timer
	streamsdk::SDKCommon * sampleCommon;    /**< SDKCommon class object */	
	TimerList*	_timeValueList;

};

  /** Returns raw offseted of the given pointer.
    @note
        The offset are in bytes, no matter what type of the pointer.
    */
    template <class T>
    static FORCEINLINE T* rawOffsetPointer(T* ptr, ptrdiff_t offset)
    {
        return (T*)((char*)(ptr) + offset);
    }

    /** Advance the pointer with raw offset.
    @note
        The offset are in bytes, no matter what type of the pointer.
    */
    template <class T>
    static FORCEINLINE void advanceRawPointer(T*& ptr, ptrdiff_t offset)
    {
        ptr = rawOffsetPointer(ptr, offset);
    }