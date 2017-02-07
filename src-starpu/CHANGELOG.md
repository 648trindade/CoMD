## parallel.c
 - Inicialização do starpu em `initParallel`.
 - Finalização do starpu em `destroyParallel`.

## linkcells.c
 - Alocação da matriz `nbrBoxes` de forma contígua.

## memUtils.h
 - substituição de `malloc` por `starpu_malloc` em `comdMalloc`.
 - substituição de `free` por `starpu_free` em `comdFree`.

## starpu_code(.h)
 - Declaração de funções novas definidas dentro de **ljForce.c** (motivo: não modificar **ljForce.h**).
 - Declaração de *handle*s usados.
 - Declaração e definição de *codelet*s.
 - Declaração e definição de filtro pra particionamento dos dados.

## lJForce.c
 - Isolamento do laço principal para dentro de uma função `cpu_func` com assinatura seguindo especificações das *task* do starpu.
 - Registro de *handle*s para os vetores usados no laço principal.
 - Criação de *handle*s para a variável `ePot`, a qual sofrerá redução
 - Definição de funções para inicialização de cópias de `ePot` e sua redução (soma).
 - Configuração das funções e *codelet*s dos itens anteriores para que seja realizada a redução de `ePot` (função `starpu_data_set_reduction_methods`)
 - Cópia de valores usados no laço para uma estrutura `params`.
 - Particionamento dos vetores/matrizes `s->atoms->f`, `s->atoms->U` e `s->boxes->nbrBoxes`.