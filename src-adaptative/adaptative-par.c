#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>

#define MIN(a,b) ((a<=b)?a:b)
#define MAX(a,b) ((a<=b)?b:a)
#define ALPHA 1
#define LOG2(X) (63 - __builtin_clzll(X))

// Estrutura representando um intervalo
typedef struct{
    size_t f; // Início do intervalo
    size_t l; // Fim do intervalo
    size_t m; // Tamanho do intervalo
    size_t c; // Tamanho do chunk
} adpt_range_t;

typedef struct {
    size_t id;          // ID desta thread
    adpt_range_t range; // intervalo
    omp_lock_t lock;    // trava
    char* visited;      // vetor de histórico de acesso
} worker_data_t;

int count = 0;

void initialize_worker_data(worker_data_t* wrkr_d, size_t thr_id, size_t nthr, size_t first, size_t last){
    wrkr_d->id = thr_id; // ID desta thread
    wrkr_d->visited = malloc(nthr); // Vetor de vetores de histórico de acesso

    size_t chunk = (last - first) / nthr; // Tamanho da tarefa inicial (divisão inteira)
    size_t remain = (last - first) % nthr; // Sobras da divisão anterior
    wrkr_d->range.f = chunk * thr_id + first + MIN(thr_id, remain); // Ponto inicial do intervalo
    wrkr_d->range.l = wrkr_d->range.f + chunk + (size_t)(thr_id < remain); // Ponto final
    wrkr_d->range.m = wrkr_d->range.l - wrkr_d->range.f;  // Tamanho original do intervalo
    wrkr_d->range.c = LOG2(wrkr_d->range.m); // Tamanho do chunk a ser extraído serialmente

    omp_init_lock(&(wrkr_d->lock)); // Inicializa a trava
}

void destroy_worker_data(worker_data_t* wrkr_d){
    free(wrkr_d->visited);
    omp_destroy_lock(&(wrkr_d->lock));
}

/*
 * Function: adpt_extract_seq
 * --------------------------
 *   Extrai trabalho sequencial do intervalo pertencente a thread corrente.
 *   A quantia de trabalho a ser extraída é ALPHA * log2(m), onde m é o tamanho do intervalo.
 *   Caso detecte um conflito (algum ladrão rouba parte do trabalho que ele 
 *   extraíria), tenta extrair novamente.
 * 
 *   wrkr_d : estrutura com info. do worker
 *        f : o índice inicial do trabalho extraído
 *        l : o índice final do trabalho extraído
 * 
 *   returns : o tamanho do intervalo extraído
 */
size_t adpt_extract_seq(worker_data_t* wrkr_d, size_t* f, size_t* l){
    while(1) {
        *f = wrkr_d->range.f;
        *l = MIN(wrkr_d->range.l, *f + (size_t)(ALPHA * wrkr_d->range.c));
        wrkr_d->range.f = *l;
        if (wrkr_d->range.f > wrkr_d->range.l) { // Conflito: desfaz e tenta de novo
            wrkr_d->range.f = *f;
#ifdef ADPT_DEBUG
            fprintf(stderr, "%.6lf [Worker %d] conflitou ao extrair trabalho\n", omp_get_wtime(), wrkr_d->id);
#endif
            continue;
        }
#ifdef ADPT_DEBUG
        fprintf(stderr, "%.6lf [Worker %d] adpt_extract_seq %d a %d (%d)\n", omp_get_wtime(), omp_get_thread_num(), *f, *l, wrkr_d->range.l);
#endif
        return *l - *f;
    }
}

/*
 * Function: adpt_extract_par
 * --------------------------
 *   Rouba trabalho sequencial de um intervalo pertencente a uma outra thread.
 *   O algoritmo seleciona aleatoriamente um intervalo e faz os seguintes testes:
 *    - Se já não foi visitado
 *    - Se pode ser travado
 *    - Se possui a quantia de trabalho necessária para roubo
 *   A quantia de trabalho necessária é um valor entre ((m - m')/2) e (sqrt(m)),
 *   onde m é o tamanho do intervalo e m' é o tamanho do intervalo já trabalhado.
 * 
 *    theft_d : estrutura com info. do ladrão
 *   v_wrkr_d : vetor de estruturas com info. dos workers
 *       nthr : número total de threads da aplicação
 * 
 *   returns : 1 se o roubo for bem sucedido, 0 do contrário
 */
size_t adpt_extract_par(
    worker_data_t* theft_d, worker_data_t* v_wrkr_d, size_t nthr
){
    size_t success     = 0;
    size_t remaining   = nthr - 1;
    size_t i;

    memset(theft_d->visited, 0, nthr);
    theft_d->visited[theft_d->id] = 1;

    // Enquanto houver intervalos que não foram inspecionadas
    while((remaining > 0) && !success){
        i = rand() % nthr; // Sorteia uma intervalo
        for (; theft_d->visited[i]; i = (i+1)%nthr); // Avança até um intervalo ainda não visto
        worker_data_t* victim_d = &(v_wrkr_d[i]);

        // Testa se o intervalo não está travado e trava se estiver livre.
        if (omp_test_lock(&(victim_d->lock))){
            size_t vic_l = victim_d->range.l, vic_f = victim_d->range.f; // cópias
            size_t m_left = vic_l - vic_f; // Trabalho restante
            size_t min = (size_t)sqrt(victim_d->range.m); // Tamanho mínimo para roubo
            size_t steal_size = MAX(m_left >> 1, min); // Tamanho do roubo
            size_t begin = vic_l - steal_size;

            // O restante é maior ou igual ao mínimo a ser roubado?
            if ((m_left > min) && (steal_size > 1) && (vic_f < vic_l)){
                victim_d->range.l = begin;
                if (victim_d->range.f < begin){ // Inicio a frente do inicio da vitima
                    // ativa a própria trava pra ninguém roubar dele enquanto ele rouba
                    omp_set_lock(&(theft_d->lock));
                    // Efetua o roubo, atualizando os intervalos
                    theft_d->range.f  = begin;
                    theft_d->range.l  = vic_l;
                    theft_d->range.m  = steal_size;
                    omp_unset_lock(&(theft_d->lock));
                    success = 1;
                }
                else{ // Conflito: desfaz e aborta
                    victim_d->range.l = vic_l;
#ifdef ADPT_DEBUG
                    fprintf(stderr, "%.6lf [Worker %d] conflitou ao roubar de %lu\n", omp_get_wtime(), theft_d->id, i);
#endif
                }
            }
            else { // Caso não houver trabalho suficiente
                theft_d->visited[i] = 1; // Marca como visitado
                --remaining;      // Decrementa o total de intervalos restantes para inspeção
            }
            omp_unset_lock(&(victim_d->lock)); // Destrava o intervalo
        }
    }
#ifdef ADPT_DEBUG
    if (success)
        fprintf(stderr, "%.6lf [Worker %d] adpt_extract_par %d a %d de %lu\n", omp_get_wtime(), theft_d->id, theft_d->range.f, theft_d->range.l, i);
    else
        fprintf(stderr, "%.6lf [Worker %d] sem trabalho disponivel\n", omp_get_wtime(), theft_d->id);
#endif
    return success;
}

/*
 * Function: adpt_parallel_for_and_reduce
 * --------------------------------------
 * 
 *        kernel : função de processamento
 *          data : dados passados como argumentos à função
 *         first : início do laço
 *          last : fim do laço
 *   start_value : valor inicial para a redução
 *      rdct_fun : função de redução
 * 
 *   returns : o valor reduzido
 */
double adpt_parallel_for_and_reduce(
    double (*kernel)(void*, size_t), void* data, size_t first, size_t last, 
    double start_value, double (*rdct_fun)(double, double)
){
    // adpt_range_t* ranges;  // Vetor de intervalos (uma por thread)
    // omp_lock_t*   locks;   // Vetor de travas (uma por thread)
    // char*         visited; // Vetor de vetores de histórico de acesso (1 por thread)
    // size_t           nthr;    // Número de threads
    // double        rdct_var = start_value; // Variável de redução

    // nthr    = omp_get_max_threads();
    // ranges  = malloc(nthr * sizeof(adpt_range_t));
    // locks   = malloc(nthr * sizeof(omp_lock_t));
    // visited = malloc(nthr * nthr);

    size_t nthr = omp_get_max_threads(); // número de threads
    worker_data_t* workers_data = malloc(nthr * sizeof(worker_data_t)); // Dados para os workers
    double rdct_var = start_value; // Variável de redução

    // fprintf(stderr, "%.6lf ============ STARTING ==============\n", omp_get_wtime());
    #pragma omp parallel shared(kernel, data, start_value, rdct_fun, workers_data, nthr, first, last, rdct_var)
    {
        // size_t           thr_id     = omp_get_thread_num();      // ID desta thread
        // adpt_range_t* m_range    = &ranges[thr_id];           // intervalo desta thread
        // omp_lock_t*   m_lock     = &locks[thr_id];            // Trava desta thread
        // char*         m_visited  = visited + (thr_id * nthr); // Vetor de histórico desta thread
        // size_t           chunk      = (last - first) / nthr;     // Tamanho da tarefa inicial
        // double        m_rdct_var = start_value;
        
        // m_range->f = chunk * thr_id + first;        // Cálculo do ponto inicial do intervalo
        // m_range->l = MIN(m_range->f + chunk, last); // Cálculo do ponto final do intervalo
        // m_range->m = m_range->l - m_range->f;       // Cálculo do tamanho original do intervalo

        // omp_init_lock(m_lock); // Inicializa a trava

        size_t m_first = first, m_last = last; // cópia dos limites originais
        double m_rdct_var = start_value;
        size_t thr_id = omp_get_thread_num();  // ID desta thread
        worker_data_t* m_worker_data = &workers_data[thr_id]; // pega o worker desta thread
        initialize_worker_data(m_worker_data, thr_id, nthr, first, last); // inicia o worker
        #pragma omp barrier

        while(1){ // Itera enquanto houver trabalho a ser feito
            // Itera enquanto houver trabalho sequencial a ser feito
            while (adpt_extract_seq(m_worker_data, &m_first, &m_last) > 0)
                for (size_t i = m_first; i < m_last; i++)
                    // Chama o kernel da aplicação e aplica redução no resultado retornado
                    m_rdct_var = rdct_fun(m_rdct_var, kernel(data, i));
            
            // Tenta roubar trabalho, se não conseguir, sai do laço
            if (!adpt_extract_par(m_worker_data, workers_data, nthr))
                break;
        }

        #pragma omp critical
        rdct_var = rdct_fun(rdct_var, m_rdct_var); // Reduz os resultados por thread

        #pragma omp barrier
        destroy_worker_data(m_worker_data); // Destrói a trava
    }

    free(workers_data);
    return rdct_var;
}

/*
 * Function: adpt_parallel_for
 * ---------------------------
 * 
 *   kernel : função de processamento
 *     data : dados passados como argumentos à função
 *    first : início do laço
 */
void adpt_parallel_for(
    void (*kernel)(void*, size_t), void* data, size_t first, size_t last
){
    size_t nthr = omp_get_max_threads(); // número de threads
    worker_data_t* workers_data = malloc(nthr * sizeof(worker_data_t)); // Dados para os workers

#ifdef ADPT_DEBUG
    fprintf(stderr, "%.6lf ============ STARTING ==============\n", omp_get_wtime());
#endif
    #pragma omp parallel shared(kernel, data, workers_data, nthr, first, last)
    {
        size_t m_first = first, m_last = last; // cópia dos limites originais
        size_t thr_id = omp_get_thread_num();  // ID desta thread
        worker_data_t* m_worker_data = &workers_data[thr_id]; // pega o worker desta thread
        initialize_worker_data(m_worker_data, thr_id, nthr, first, last); // inicia o worker
        #pragma omp barrier

        while(1){ // Itera enquanto houver trabalho a ser feito
            // Itera enquanto houver trabalho sequencial a ser feito
            while (adpt_extract_seq(m_worker_data, &m_first, &m_last) > 0)
                for (size_t i = m_first; i < m_last; i++)
                    kernel(data, i);
            
            // Tenta roubar trabalho, se não conseguir, sai do laço
            if (!adpt_extract_par(m_worker_data, workers_data, nthr))
                break;
        }

        #pragma omp barrier
        destroy_worker_data(m_worker_data); // Limpa o worker
    }
    free(workers_data);

    if (count){
        printf("Counter: %d\n", count);
        count = 0;
    }
}