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
};