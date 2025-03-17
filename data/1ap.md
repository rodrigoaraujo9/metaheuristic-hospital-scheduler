# Representação de Estado no Problema de Alocação de Pacientes

## 1. Estrutura de Dados de Entrada

### 1.1 Pacientes

Suponha que haja *P* pacientes. Para cada paciente *i*, mantemos os seguintes dados:

- *Janela de admissão:*
  - *Ei* (earliest admission): dia mais cedo em que pode ser admitido.
  - *Li* (latest admission): dia mais tarde em que pode ser admitido.
- *Duração da internação:* *LOSi*. Define quantos dias o paciente fica internado após ser admitido.
- *Duração de cirurgia:* *dcirurgia(i)*. Se a cirurgia ocorre no dia de admissão, esse valor será usado no cálculo de consumo de bloco cirúrgico. (Pode ser nulo para pacientes sem cirurgia.)
- *Especialização principal:* *Speci. Indica se o paciente requer **THO, ABD, ONC, URO* etc.
- *Workload diário:* *Workloadi,1...LOSi*. Vetor descrevendo quanta carga de trabalho o paciente gera em cada dia da internação (ex.: [1.2, 1.13, 1.0, ...]).
- *(Opcional) Tipo de paciente:* Se é eletivo ou emergencial (pode influenciar a prioridade de alocação, se houver).

### 1.2 Wards

Suponha que existam *W* wards (setores ou alas hospitalares). Cada ward *w* possui:

- *Capacidade de leitos diária:* *Bedsw,d* para cada dia *d. Indica o número máximo de pacientes simultâneos no dia **d*.
- *Especialização major:* *Specwmajor*. Se um paciente tiver essa especialização como principal, não há fator extra de carga.
- *Especializações minor:* *Specwminor*. Se o paciente tiver uma especialização que está neste conjunto, o workload pode receber um fator de escala (menos eficiência).
- *(Opcional) Carryover:* *CarryOverw,d*. Número de pacientes (e workload) herdados de períodos anteriores, já ocupando leitos no início.

### 1.3 Especializações e Bloco Cirúrgico

Para cada especialização *s*, temos:

- *Capacidade de bloco cirúrgico (OT) diária:* *OTs,d. É o total de minutos (ou unidade de tempo) disponível para cirurgias da especialização **s* no dia *d*.
- *Fator de escalonamento para wards* em que *s* seja considerada minor. Esse valor multiplicará parte do workload diário se o paciente for de especialização *s* e estiver num ward em que *s* não seja major.

## 2. Representação da Solução (Estado)

Uma solução é, basicamente, a atribuição de cada paciente *i* a:

- *Dia de admissão:* *ai, no intervalo **[Ei, Li]*.
- *Ward:* *wi*.
- *(Opcional) Dia de cirurgia:* *ci*, se for separado da admissão.

### Exemplo de Estrutura em Python

```python
# Caso cirurgia = admissão
solution = [
    (a_0, w_0),    # paciente 0
    (a_1, w_1),    # paciente 1
    ...
    (a_{P-1}, w_{P-1})
]

# Caso dia de cirurgia seja separado
solution = [
    (a_0, w_0, c_0),
    (a_1, w_1, c_1),
    ...
    (a_{P-1}, w_{P-1}, c_{P-1})
]
```

## 3. Restrições Principais

### 3.1 Janela de admissão

Ei ≤ ai ≤ Li, ∀i. Se isso não for respeitado, a solução é inválida.

### 3.2 Capacidade de leitos

Para cada dia d e ward w, somamos quantos pacientes i estão internados ali nesse dia. A soma não pode exceder Bedsw,d.

### 3.3 Capacidade do bloco cirúrgico (OT)

Se a cirurgia ocorre no dia de admissão ai:


Soma das cirurgias no dia ai ≤ Capacidade do bloco cirúrgico


Se a cirurgia ocorre em ci, basta trocar ai por ci no somatório.

---

## 4. Função Objetivo (Cost Function)

Para minimizar penalidades ou custos, podemos combinar:

### 4.1 Atraso de admissão

Minimizar o tempo de espera entre a data desejada e a data real de admissão.

### 4.2 Overtime do Bloco Cirúrgico

Minimizar o tempo extra utilizado no bloco cirúrgico.

### 4.3 Undertime do Bloco Cirúrgico

Minimizar o tempo ocioso do bloco cirúrgico.

### 4.4 Balanceamento de carga (opcional)

Distribuir as admissões e cirurgias de forma equilibrada ao longo do tempo.

### 4.5 Exemplo de Função Objetivo Geral

Combina as penalidades das seções anteriores com pesos apropriados.

---

## 5. Observações Finais

Essa modelagem unifica a representação para meta-heurísticas como hill-climbing, simulated annealing, tabu search e algoritmos genéticos.

- Em busca local, a vizinhança pode alterar (ai, wi).
- Em algoritmos genéticos, cada cromossomo é um vetor de (ai, wi, ci).