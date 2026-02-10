# Mecânica do Voo — Simulador 6DOF EMB-110 Bandeirante

Repositório contendo a implementação de um **simulador de dinâmica de voo 6DOF**
para análise da **estabilidade dinâmica da aeronave EMB-110 Bandeirante**.

O projeto foi desenvolvido como trabalho da disciplina **Mecânica do Voo** e
tem como foco a modelagem física, aerodinâmica e a simulação numérica do
comportamento dinâmico da aeronave.

---

##  Modelo Físico e Matemático

A modelagem matemática é formulada em variáveis de estado conforme descrito em  
**Aircraft Flight Dynamics and Control — Wayne Durham**.

Hipóteses adotadas:
- Massa constante
- Aeronave indeformável (hipótese de corpo rígido)
- Aproximação de Terra plana  
  (efeitos de curvatura, Coriolis e aceleração centrífuga desprezados)

O modelo considera as **12 equações diferenciais do movimento** de uma aeronave
em seis graus de liberdade.

---

##  Modelo Aerodinâmico

A modelagem aerodinâmica foi realizada a partir de duas abordagens complementares:

- **Athena Vortex Lattice (AVL)**  
  - Aerodinâmica linear em regime estacionário  
  - Avaliação de diferentes posições do centro de gravidade  
  - Extração das derivadas aerodinâmicas

- **XFLR5 (Método dos Painéis)**  
  - Análise dos perfis aerodinâmicos da aeronave  
  - Obtenção das curvas polares dos aerofólios utilizados

---

##  Solução Numérica

A simulação considera uma condição inicial de **voo reto e nivelado**, adotando o
modelo de **pequenas perturbações**.

- O vetor de estado inicial é fornecido ao integrador numérico
- As 12 equações diferenciais são resolvidas pelo método de  
  **Runge–Kutta de quarta ordem**
- É possível aplicar:
  - Entradas **degrau**
  - Entradas **doublet**
  nas superfícies de controle, permitindo a análise da resposta dinâmica no tempo

---

##  Dados e Estrutura do Repositório

- **Aerofólios (`.dat`)**  
  Coordenadas dos aerofólios utilizados, parametrizadas pela corda

- **Arquivos `.avl`**  
  Modelos geométricos da aeronave utilizados no Athena Vortex Lattice,
  com variação da posição do centro de gravidade

- **Arquivos `.mass`**  
  Dados de massa e inércia utilizados no AVL, considerando diferentes
  configurações de massa e centro de gravidade

- **Códigos Python**  
  Implementação do simulador e notebooks auxiliares para cálculo dos
  parâmetros da aeronave

- **Derivadas aerodinâmicas**  
  Derivadas obtidas no Athena Vortex Lattice para cada caso avaliado

- **Polares dos aerofólios**  
  Curvas polares obtidas no XFLR5 para cada aerofólio analisado
