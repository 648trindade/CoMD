#ifndef ADAPTATIVEPAR_H_
#define ADAPTATIVEPAR_H_

double adpt_parallel_for_and_reduce(
    double (*kernel)(void*, int, int), void* data, int first, int last, 
    double start_value, double (*rdct_fun)(double, double)
);

void adapt_parallel_for(
    void (*kernel)(void*, int, int), void* data, int first, int last
);

#endif //ADAPTATIVEPAR_H_