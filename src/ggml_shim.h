#ifndef GGML_SHIM_H
#define GGML_SHIM_H

#include "ggml.h"

#define GGML_VERSION 100
#define GGML_COMMIT "unknown"

struct ggml_compute_params {
    int type; // enum ggml_task_type
    int ith;
    int nth;
    void * wsize;
    void * wdata;
};

#endif

