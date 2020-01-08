#pragma  once 
#ifndef FUNCTION_OTHERS_CUH
#define FUNCTION_OTHERS_CUH
#include <iostream>
#include <cudnn.h>


extern void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w);

extern void get_path(std::string& sFilename, const char *fname, const char *pname);

#endif
