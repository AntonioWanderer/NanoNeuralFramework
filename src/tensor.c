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

