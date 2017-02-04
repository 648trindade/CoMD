## parallel.c
 - Inicialização do starpu em `initParallel`
 - Finalização do starpu em `destroyParallel`

## memUtils.h
 - substituição de `malloc` por `starpu_malloc` em `comdMalloc`
 - substituição de `free` por `starpu_free` em `comdFree`

## starpu_code(.h)
 - Declaração de funções novas definidas dentro de **ljForce.c** (motivo: não modificar **ljForce.h**)

## lJForce.c
 - Isolamento do laço principal para dentro de uma função `cpu_func` com assinatura seguindo especificações das *task* do starpu.
 - Criação de *handle*s para os vetores usados no laço principal.
 - Particionamento da matriz com os números dos átomos vizinhos.
 - Criação de *handle*s para a variável `ePot`, a qual sofrerá redução
 - Definição de funções para inicialização de cópias de `ePot` e sua redução (soma).
 - Criação de *codelet*s para as funções ditas acima
 - Configuração das funções e *codelet*s dos itens anteriores para que seja realizada a redução de `ePot` (função `starpu_data_set_reduction_methods`)
 - Cópia de valores usados no laço para uma estrutura `params`
 - Reuso do mesmo codelet pra todas as *task*s criadas.
 - Criação de uma *task* por *worker*, com uma cópia própria de `params` e um pedaço específico do vetor dos vizinhos. (`s->boxes->nbrBoxes`).