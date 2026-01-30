#include "../include/tensor.h"
#include <stdio.h>
#include <time.h>

void test_creation() {
    printf("===Test Tensor Creation===");
    
    int shape1[] = {5, 3};
    Tensor* tensor1 = tensor_ones(shape1, 2, DTYPE_FLOAT32);
    tensor_print(tensor1);
    tensor_free(tensor1);

    int shape2[] = {5, 3, 4, 7};
    Tensor* tensor2 = tensor_ones(shape2, 4, DTYPE_FLOAT32);
    tensor_print(tensor2);
    tensor_free(tensor2);

    printf("OK");
}

void test_matmul_small() {
    printf("===Test Tensor Matmul===");

    int shape1[] = {2, 3};
    float data1[] = {4, 2, 7, 3, 10, 1};
    int shape2[] = {3, 2};
    float data2[] = {7, 8, 2, 9, 1, 5};

    float expected_result[] = {39, 85, 42, 119};

    Tensor* tensor1 = tensor_from_data(data1, shape1, 2, DTYPE_FLOAT32);
    Tensor* tensor2 = tensor_from_data(data2, shape2, 2, DTYPE_FLOAT32);

    Tensor* result = tensor_matmul_naive(tensor1, tensor2);
    

    assert(result->data);
    
    tensor_free(tensor1);
    tensor_free(tensor2);
    tensor_free(result);
}

void test_relu() {
    
}