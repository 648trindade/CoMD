#ifndef ADAPTATIVEPAR_H_
#define ADAPTATIVEPAR_H_

#include <stddef.h>

double adpt_parallel_for_and_reduce(
    double (*kernel)(void*, size_t), void* data, size_t first, size_t last, 
    double start_value, double (*rdct_fun)(double, double)
);

void adpt_parallel_for(
    void (*kernel)(void*, size_t), void* data, size_t first, size_t last, FILE* trace_file
);

#endif //ADAPTATIVEPAR_H_