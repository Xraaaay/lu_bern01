# Project 1

> Markov Chain Monte Carlo(MCMC) Based Modeling

## Structure

- `main.py`: setup, execution

- `mcmc.py`: Monte Carlo sampling algorithm

- `model.py`: 2D system class

## Background

### Monte Carlo

Random Variable $X$ follows a PDF $P(x)$. For a function $R(x)$, we can do the **importance sampling**. The expectation: $$<R> = \frac{1}{N} \sum_{i=1}^N R(x_i)$$, where $x_i$ is the $i$th draw from $P(x)$.

### Markov Chain (TBD)

> We use Markov Chain to generate the target distribution $P(x)$.

transition matrix $T$

stationary distribution $\boldsymbol{\rho}$

detailed balance: $$T_{mn} \rho_n = T_{nm} \rho_m$$

### Metropolis-Hastings Algorithm (TBD)

> Metropolis algorithm is an implementation of Markov Chain.

state $\rho_m \to \rho_n$:

- Proposal Probability $\alpha_{nm}$

- Accpectance Probability $P(m \to n)$

The detaild balance equation $$\alpha_{nm} P(m \to n) \rho_m = \alpha_{mn} P(n \to m) \rho_n$$ holds if we take $$P(m \to n) = \min (1, \frac{\alpha_{mn} \rho_n}{\alpha_{nm} \rho_m})$$

The proposal probability is taken to be symmetric ($\alpha_{nm} = \alpha_{mn}$) in Metropolis algorithm.

### Boltzmann Distribution

> Markov Chain Monte Carlo can be used to estimate the expectated energy of a system following the Boltzmann Distribution.

Boltzmann Distribution: $$P_B(\boldsymbol{r}^N) = \frac{1}{Z} e^{-\beta U(\boldsymbol{r}^N)}, Z = \sum_{\boldsymbol{r}^N} e^{-\beta U(\boldsymbol{r}^N)}$$, where $\displaystyle \beta = \frac{1}{k_B T}$.

The interaction energy: $$U(\boldsymbol{r}^N) = \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \Phi (\boldsymbol{r}_i, \boldsymbol{r}_j)$$

By importance Monte Carlo sampling, we can estimate the average for the interaction energy: $$<U> = \frac{1}{M} \sum_{i=1}^M U(\boldsymbol{r}_i^N)$$
