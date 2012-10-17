#pragma once

#define SIZE_ARRAY		3
#define SIZE_BUFFER	10

class PointerArray
{
public:
	PointerArray();
	~PointerArray();

	void initialize();
	void unInitialize();

protected:
private:
	float *_pointerArray[SIZE_ARRAY];
};