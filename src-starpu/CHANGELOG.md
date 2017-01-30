## parallel.c
 - Inicialização do starpu em `initParallel`
 - Finalização do starpu em `destroyParallel`

## memUtils.h
 - substituição de `malloc` por `starpu_malloc` em `comdMalloc`
 - substituição de `free` por `starpu_free` em `comdFree`

## starpu_code(.c/.h)
 - Declaração de funções novas definidas dentro de **ljForce.c** (motivo: não modificar **ljForce.h**)
 - Definição e declaração de funções para criar, registrar, desregistrar e apagar *handle*s para os buffers usados pela lJForce

## lJForce.c
 - Isolamento do laço principal para dentro de uma função `cpu_func` com assinatura seguindo especificações das *task* do starpu.
 - Criação de *handle*s para os vetores usados no laço principal.
 - Criação de *handle*s para a variável `ePot`, a qual sofrerá redução
 - Definição de funções para inicialização de cópias de `ePot` e sua redução (soma).
 - Criação de *codelet*s para as funções ditas acima
 - Configuração das funções e *codelet*s dos itens anteriores para que seja realizada a redução de `ePot` (função `starpu_data_set_reduction_methods`)
 - Cópia de valores usados no laço para uma estrutura `params`
 - Criação de um codelet para cada *task* submetida
 - Criação de uma *task* por laço, a qual tem um *codelet* e um `params` específico.