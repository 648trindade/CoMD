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
    int f; // Início do intervalo
    int l; // Fim do intervalo
    int m; // Tamanho do intervalo
} adpt_range_t;

/*
 * Function: adpt_extract_seq
 * --------------------------
 *   Extrai trabalho sequencial do intervalo pertencente a thread corrente.
 *   A quantia de trabalho a ser extraída é ALPHA * log2(m), onde m é o tamanho do intervalo.
 * 
 *   m_range : ponteiro para o intervalo da thread corrente
 *         f : o índice inicial do trabalho extraído
 *         l : o índice final do trabalho extraído
 * 
 *   returns : o tamanho do intervalo extraído
 */
int adpt_extract_seq(adpt_range_t* m_range, int* f, int* l){
    *f = m_range->f;
    *l = MIN(m_range->l, *f + (int)(ALPHA * LOG2(m_range->m)));
    m_range->f = *l;
    // fprintf(stderr, "%.6lf [Worker %d] adpt_extract_seq %d a %d\n", omp_get_wtime(), omp_get_thread_num(), *f, *l);
    return *l - *f;
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
 *      ranges : vetor de intervalos
 *     m_range : intervalo da thread corrente
 *       locks : vetor de travas
 *   m_visited : vetor auxiliar que armazena os intervalos que já foram inspecionados
 *        nthr : número total de threads da aplicação
 *      thr_id : ID referente a thread corrente
 * 
 *   returns : 1 se o roubo for bem sucedido, 0 do contrário
 */
int adpt_extract_par(
    adpt_range_t* ranges, adpt_range_t* m_range, 
    omp_lock_t* locks, char* m_visited, 
    int nthr, int thr_id
){
    int success     = 0;
    int remaining   = nthr - 1;

    memset(m_visited, 0, nthr);
    m_visited[thr_id] = 1;

    // Enquanto houver intervalos que não foram inspecionadas
    while((remaining > 0) && !success){
        int i = rand() % nthr; // Sorteia uma intervalo
        while (m_visited[i] != 0) i = (i+1) % nthr; // Avança até um intervalo ainda não visto

        // Testa se o intervalo não está travado e trava se estiver livre.
        // omp_test_lock é uma chamada não-bloqueante
        if (omp_test_lock(&locks[i])){
            int m_left = ranges[i].l - ranges[i].f;  // Calcula o trabalho restante
            int min    = sqrt(ranges[i].m);          // Calcula o tamanho mínimo para roubo
            int steal_size  = MAX(m_left >> 1, min); // Calcula o tamanho do roubo

            // Testa se o restante é maior ou igual ao mínimo a ser roubado
            // FIX: não roubar se falta só um.
            if ((m_left >= min) && (steal_size > 1)){
                // Efetua o roubo, atualizando os intervalos
                m_range->f  = ranges[i].l - steal_size;
                m_range->l  = ranges[i].l;
                ranges[i].l = m_range->f;
                m_range->m  = m_range->l - m_range->f;
                
                // fprintf(stderr, "%.6lf [Worker %d] adpt_extract_par %d a %d\n", omp_get_wtime(), thr_id, m_range->f, m_range->l);
                success = 1;
            }
            else { // Caso não houver trabalho suficiente
                m_visited[i] = 1; // Marca como visitado
                --remaining;      // Decrementa o total de intervalos restantes para inspeção
            }
            omp_unset_lock(&locks[i]); // Destrava o intervalo
        }
    }
    // if (!success)
    //     fprintf(stderr, "%.6lf [Worker %d] sem trabalho disponivel\n", omp_get_wtime(), thr_id);
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
    double (*kernel)(void*, int, int), void* data, int first, int last, 
    double start_value, double (*rdct_fun)(double, double)
){
    adpt_range_t* ranges;  // Vetor de intervalos (uma por thread)
    omp_lock_t*   locks;   // Vetor de travas (uma por thread)
    char*         visited; // Vetor de vetores de histórico de acesso (1 por thread)
    int           nthr;    // Número de threads
    double        rdct_var = start_value; // Variável de redução

    nthr    = omp_get_max_threads();
    ranges  = malloc(nthr * sizeof(adpt_range_t));
    locks   = malloc(nthr * sizeof(omp_lock_t));
    visited = malloc(nthr * nthr);

    // fprintf(stderr, "%.6lf ============ STARTING ==============\n", omp_get_wtime());
    #pragma omp parallel shared(data, ranges, locks, visited, nthr, rdct_var) firstprivate(first, last)
    {
        int           thr_id     = omp_get_thread_num();      // ID desta thread
        adpt_range_t* m_range    = &ranges[thr_id];           // Intervalo desta thread
        omp_lock_t*   m_lock     = &locks[thr_id];            // Trava desta thread
        char*         m_visited  = visited + (thr_id * nthr); // Vetor de histórico desta thread
        int           chunk      = (last - first) / nthr;     // Tamanho da tarefa inicial
        double        m_rdct_var = start_value;
        
        m_range->f = chunk * thr_id + first;        // Cálculo do ponto inicial do intervalo
        m_range->l = MIN(m_range->f + chunk, last); // Cálculo do ponto final do intervalo
        m_range->m = m_range->l - m_range->f;       // Cálculo do tamanho original do intervalo

        omp_init_lock(m_lock); // Inicializa a trava
        #pragma omp barrier

        // Itera enquanto houver trabalho a ser feito
        while(1){
            // Itera enquanto houver trabalho sequencial a ser feito
            while (adpt_extract_seq(m_range, &first, &last) > 0)
                // Chama o kernel da aplicação e aplica redução no resultado retornado
                m_rdct_var = rdct_fun(m_rdct_var, kernel(data, first, last));
            
            // Tenta roubar trabalho, se não conseguir, sai do laço
            if (!adpt_extract_par(ranges, m_range, locks, m_visited, nthr, thr_id))
                break;
        }

        #pragma omp critical
        rdct_var = rdct_fun(rdct_var, m_rdct_var); // Reduz os resultados por thread

        #pragma omp barrier
        omp_destroy_lock(m_lock); // Destrói a trava
    }

    free(ranges);
    free(visited);
    free(locks);
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
    void (*kernel)(void*, int, int), void* data, int first, int last
){
    adpt_range_t* ranges;  // Vetor de intervalos (uma por thread)
    omp_lock_t*   locks;   // Vetor de travas (uma por thread)
    char* visited;         // Matriz de histórico de acesso (1 linha por thread)
    int   nthr;            // Número de threads

    nthr    = omp_get_max_threads();
    ranges  = malloc(nthr * sizeof(adpt_range_t));
    locks   = malloc(nthr * sizeof(omp_lock_t));
    visited = malloc(nthr * nthr);

    #pragma omp parallel shared(data, ranges, locks, visited, nthr) firstprivate(first, last)
    {
        int           thr_id     = omp_get_thread_num();      // ID desta thread
        adpt_range_t* m_range    = &ranges[thr_id];           // Intervalo desta thread
        omp_lock_t*   m_lock     = &locks[thr_id];            // Trava desta thread
        char*         m_visited  = visited + (thr_id * nthr); // Vetor de histórico desta thread
        int           chunk      = (last - first) / nthr;     // Tamanho da tarefa inicial
        
        m_range->f = chunk * thr_id + first;        // Cálculo do ponto inicial do intervalo
        m_range->l = MIN(m_range->f + chunk, last); // Cálculo do ponto final do intervalo
        m_range->m = m_range->l - m_range->f;       // Cálculo do tamanho original do intervalo

        omp_init_lock(m_lock); // Inicializa a trava
        #pragma omp barrier

        // Itera enquanto houver trabalho a ser feito
        while(1){
            // Itera enquanto houver trabalho sequencial a ser feito
            while (adpt_extract_seq(m_range, &first, &last) > 0)
                kernel(data, first, last);
            
            // Tenta roubar trabalho, se não conseguir, sai do laço
            if (!adpt_extract_par(ranges, m_range, locks, m_visited, nthr, thr_id))
                break;
        }

        #pragma omp barrier
        omp_destroy_lock(m_lock); // Destrói a trava
    }

    free(ranges);
    free(visited);
    free(locks);
}