#include <THC/THC.h>
#ifdef _WIN32
#include <ATen/ATen.h>
#endif
#include <stdio.h>
#include "nms_cuda_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
//
#ifdef _WIN32
//pytorch 0.4.1
//THCState *state = at::globalContext().getTHCState();
//pytorch 0.4.0
THCState *state = at::globalContext().thc_state;
#else
extern THCState *state;
#endif

extern "C"{
int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
		     THCudaIntTensor *num_out, float nms_overlap_thresh) {

    nms_cuda_compute(THCudaIntTensor_data(state, keep_out),
                     THCudaIntTensor_data(state, num_out),
                     THCudaTensor_data(state, boxes_host),
                     THCudaTensor_size(state, boxes_host, 0),
                     THCudaTensor_size(state, boxes_host, 1),
                     nms_overlap_thresh);

	return 1;
}
}
