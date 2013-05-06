#pragma once

#include "CL\cl.h"
#include "Vertex.h"
#include "Joint.h"

class CMatrixMulVector
{
public:
	CMatrixMulVector();
	~CMatrixMulVector();

public:
	// ���ݳ�ʼ�������ꡢ����
	void initialize(int sizeVertex, int sizeJoint);
	
	// �������٣����ꡢ����
	void unInitialize();

	// ִ�и����㷨
	void ExecuteNativeCPP();
	void ExecuteNativeSSE();
	
	// ��֤����Ƿ���ȷ
	bool verifyEqual();

	void SetupKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue);
	void SetupWorksize(size_t* globalWorkSize, size_t* localWorkSize, int dim);
	bool ExecuteKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue);

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
	cl_mem g_pfOCLIndex ;
	cl_mem g_pfOCLMatrix ;

	cl_context	_context ;
	cl_device_id _device_ID ;
	cl_command_queue _cmd_queue ;
	cl_kernel	_kernel;
};