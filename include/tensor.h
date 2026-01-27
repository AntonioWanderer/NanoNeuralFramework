#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

// ======Define Datatypes======

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

// ======Create and destroy======

Tensor* tensor_create(const int* shape, int ndim, DataType dtype);
Tensor* tensor_zeros(const int* shape, int ndim, DataType dtype);
Tensor* tensor_ones(const int* shape, int ndim, DataType dtype);
Tensor* tensor_from_data(void* data, const int* shape, int ndim, DataType dtype);

void tensor_free(Tensor* tensor);

// ======Info======

size_t tensor_size(const Tensor* tesor);
size_t dtype_size(DataType dtype);
void tensor_print(Tensor* tensor);

// ======Data access======

float* tensor_get_data_float(const Tensor* tensor);
void* tensor_get_item(const Tensor* tensor, const int* indexes);
void tensor_set_item(Tensor* tensor, const int* indexes, void* value);

// ======Math operations======

Tensor* tensor_add(const Tensor* first, const Tensor* second);
Tensor* tensor_mul(const Tensor* first, const Tensor* second);
Tensor* tensor_matmul_naive(const Tensor* first, const Tensor* second);
Tensor* tensor_relu(const Tensor* tensor);

// ======Optimized operations======

Tensor* tensor_matmul_simd(const Tensor* first, const Tensor* second);
Tensor* tensor_conv2d_naive(const Tensor* input, const Tensor* kernel);

#endif