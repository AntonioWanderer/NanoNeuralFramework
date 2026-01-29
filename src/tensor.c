#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// ======Utils methods for inner usage only======

// Calculate how many items can tensor contain
static size_t calculate_size(const int* shape, int ndim) {
    int full_size = 1;
    for (int dim_number=0; dim_number < ndim; dim_number++) {
        full_size *= shape[dim_number];
    }
    return full_size;
}

// calculate strides per linear memory to search exact items in multidim tensor
static void calculate_strides(const int* shape, int ndim, int* strides) {
    strides[ndim - 1] = 1;
    for (int dim_number=ndim-2; dim_number >= 0; dim_number--) {
        strides[dim_number] = strides[dim_number + 1] * shape[dim_number + 1];
    }
}

// ======Main functions implementation======

Tensor* tensor_create(const int* shape, int ndim, DataType dtype) {
    
    // allocate memory for tensor structure
    Tensor* new_tensor = (Tensor*)malloc(sizeof(Tensor));
    // return NULL if malloc error occured
    if (!new_tensor) {
        return NULL;
    }
    
    // set new tensor attributes that no need allocate memory
    new_tensor->dtype = dtype;
    new_tensor->ndim = ndim;
    new_tensor->owns_data = true;

    //allocate memory for tensor's shape
    new_tensor->shape = (int*)malloc(ndim * sizeof(int));
    // copy shape to tensor
    memcpy(new_tensor->shape, shape, ndim * sizeof(int));

    // calculate and set size
    new_tensor->size = calculate_size(shape, ndim);

    // init strides memory block into tensor
    new_tensor->strides = (int*)malloc(ndim * sizeof(int));
    // calculate strides values inplace
    calculate_strides(shape, ndim, new_tensor->strides);

    // get item size for memory allocation
    size_t item_size = dtype_size(dtype);
    // allocate zeros for all data
    new_tensor->data = calloc(new_tensor->size, item_size);

    return new_tensor;

}

Tensor* tensor_zeros(const int* shape, int ndim, DataType dtype) {
    // just create tensor because zero-init by calloc
    Tensor* zero_tensor = tensor_create(shape, ndim, dtype);
    return zero_tensor;
}

Tensor* tensor_ones(const int* shape, int ndim, DataType dtype) {
    Tensor* ones_tensor = tensor_create(shape, ndim, dtype);
    size_t item_size = dtype_size(dtype);
    // get pointer to data
    char* tensor_data_ptr = (char*)ones_tensor->data;
    
    // for different dtypes use different type pointers and ones
    if (dtype == DTYPE_FLOAT32) {
        for (size_t item_num=0; item_num < ones_tensor->size; item_num++) {
            ((float*)ones_tensor)[item_num] = 1.0f;
        }
    } else if (dtype == DTYPE_INT32) {
        for (size_t item_num=0; item_num < ones_tensor->size; item_num++) {
            ((int*)ones_tensor)[item_num] = 1;
        }
    }

    return ones_tensor;
}

Tensor* tensor_from_data(void* data, const int* shape, int ndim, DataType dtype) {
    Tensor* tensor_from_data = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor_from_data) {
        return NULL;
    }

    tensor_from_data->ndim = ndim;
    tensor_from_data->dtype = dtype;

    // get pointer to outer data, not owns
    tensor_from_data->owns_data = false;
    tensor_from_data->data = data;

    // copy (!) shape
    tensor_from_data->shape = (int*)malloc(ndim * sizeof(int));
    memcpy(tensor_from_data->shape, shape, ndim * sizeof(int));

    tensor_from_data->size = calculate_size(shape, ndim);

    tensor_from_data->strides = (int*)malloc(ndim * sizeof(int));
    calculate_strides(shape, ndim, tensor_from_data->strides);

    return tensor_from_data;

}

void tensor_free(Tensor* tensor) {
    if (!tensor) {
        return;
    }

    free(tensor->shape);
    free(tensor->strides);

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }

    free(tensor);
}

size_t tensor_size(const Tensor* tensor) {
    if (tensor->size) return tensor->size;
    return calculate_size(tensor->shape, tensor->ndim);
}

size_t dtype_size(DataType dtype) {
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        return sizeof(float);
    case DTYPE_INT32:
        return sizeof(int);
    case DTYPE_INT8:
        return sizeof(char);
    default:
        return sizeof(float);
    }
}

void tensor_print(Tensor* tensor) {
    printf("Tensor(shape=[");
    for (int dim = 0; dim < tensor->ndim; dim++) {
        printf("%d", tensor->shape[dim]);
        if (dim < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("], dtype=%d, size=%d items\n", tensor->dtype, tensor->size);

    printf("Data inside: [");

    int maxsize_to_print = 20;
    if (tensor->size < maxsize_to_print) {
        maxsize_to_print = tensor->size;
    }

    float* data = (float*)tensor->data;
    for (int index = 0; index < maxsize_to_print; index++) {
        printf("%.2f", data[index]);
        if (index < maxsize_to_print - 1) {
            printf(", ");
        }
    } 
    printf("]\n");

}

// get data from tensor as float
float* tensor_get_data_float(const Tensor* tensor) {
    assert(tensor->dtype == DTYPE_FLOAT32);
    return (float*)tensor->data;
}

static size_t get_linear_index(const int* strides, int ndim, const int* indexes) {
    int linear_index = 0;
    for (int dim = 0; dim < ndim; dim++) {
        linear_index += indexes[dim] * strides[dim];
    }
    return linear_index;
}

void* tensor_get_item(const Tensor* tensor, const int* indexes) {
    size_t liner_index = get_linear_index(tensor->strides, tensor->ndim, indexes);
    size_t element_size = dtype_size(tensor->dtype);
    return (char*)tensor->data + element_size * liner_index;
}

void tensor_set_item(Tensor* tensor, const int* indexes, void* value) {
    size_t liner_index = get_linear_index(tensor->strides, tensor->ndim, indexes);
    size_t element_size = dtype_size(tensor->dtype);
    memcpy((char*)tensor->data + element_size * liner_index, value, element_size);
}

