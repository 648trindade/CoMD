#ifndef ADAPTATIVEPAR_H_
#define ADAPTATIVEPAR_H_

double adpt_parallel_for_and_reduce(
    double (*kernel)(void*, size_t, size_t), void* data, size_t first, size_t last, 
    double start_value, double (*rdct_fun)(double, double)
);

void adapt_parallel_for(
    void (*kernel)(void*, size_t, size_t), void* data, size_t first, size_t last
);

#endif //ADAPTATIVEPAR_H_