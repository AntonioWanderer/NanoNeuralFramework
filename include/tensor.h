#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_INT32,
    DTYPE_INT8
} DataType;

typedef struct {
    void* data; // pointer to data
    DataType dtype; // data type from enum
    int ndim; // num of tensor's dimensions
    int* shape; // tensor shape array [dim0, dim1, ...]
    int* strides; // tensor strides per each dimension
    size_t size; //size of full tensor
    bool owns_data; // flag that tensor owns current data, not using outside data
} Tensor;




#endif