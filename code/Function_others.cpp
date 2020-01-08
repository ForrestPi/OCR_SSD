#include "Function_others.h"
#include <iostream>
#include <cudnn.h>
#include "error_util.h"

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
	sFilename = (std::string("test_data/") + std::string(fname));
	return;
}


// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w));
#elif defined(ND_TENSOR_DESCRIPTOR)
	const int nDims = 4;
	int dimA[nDims] = { n,c,h,w };
	int strideA[nDims] = { c*h*w, h*w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc,
		dataType,
		4,
		dimA,
		strideA));
#else
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc,
		dataType,
		n, c,
		h, w,
		c*h*w, h*w, w, 1));
#endif
	return;
}