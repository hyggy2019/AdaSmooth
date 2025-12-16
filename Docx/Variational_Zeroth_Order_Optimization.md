# Zeroth-Order Optimization with Anisotropic Sampling

**Abstract**

> Zeroth-order (ZO) optimization is a powerful tool for solving black-box problems where gradients are unavailable or unreliable. A central challenge in ZO methods is the choice of a smoothing (sampling) parameter, which governs a fundamental bias--variance trade-off: a small radius reduces bias but increases estimator variance, while a large radius does the opposite. This trade-off is notoriously difficult to manage in practice; fixed or heuristically tuned radii often limit efficiency and robustness.
>
> In this paper, we propose *Zeroth-Order Optimization with Anisotropic Sampling*, a principled framework that learns the sampling (smoothing) distribution online. Our key idea is to reinterpret ZO optimization as *divergence-regularized policy optimization* over a Gaussian search distribution. By solving a sequence of $f$-divergence-regularized policy updates, we obtain closed-form updates for both the mean and covariance of the search distribution. This procedure, which we term *adaptive sampling*, simultaneously (i) adapts the effective smoothing level to the local geometry of the objective, (ii) implements a variance-reduced ZO gradient estimator with an automatically learned baseline, and (iii) learns a data-driven preconditioner.
>
> We instantiate our framework into a practical algorithm, denoted \AlgName{} (Adaptive Sampling Zeroth-Order Optimization), and provide theoretical guarantees for nonconvex objectives, showing that it matches standard ZO convergence rates while maintaining controlled sampling noise. Empirically, on a range of black-box tasks---including synthetic benchmarks, adversarial attacks, and hyperparameter tuning---\AlgName{} consistently outperforms state-of-the-art ZO baselines with fixed smoothing and classical adaptive search-distribution methods such as NES and CMA-ES, in terms of query efficiency, stability, and robustness.

## Introduction

## Preliminaries

This section establishes the mathematical foundations for Zeroth-Order Optimization (ZOO) and elucidates its fundamental connection to single-step Policy Optimization (PO).

**Problem Formulation.** We consider the problem of minimizing an objective function $F(\vtheta): \sR^d \to \sR$, defined as an expectation over a random variable $\xi$:
$$
\min_{\vtheta \in \sR^d} F(\vtheta) \triangleq \E_{\xi}[f(\vtheta; \xi)],
$$
where $\vtheta \in \sR^d$ represents the decision variables and $f(\vtheta; \xi)$ denotes the stochastic loss function associated with the noise source $\xi$. We operate under the *Zeroth-Order (ZO)* setting, where the analytic gradient $\nabla_{\vtheta} F(\vtheta)$ is inaccessible or computationally prohibitive. The algorithm interacts with the objective solely through a black-box oracle that returns stochastic function evaluations $f(\vtheta; \xi)$ given query points.

**Zeroth-Order Optimization via Gaussian Smoothing.** A dominant paradigm in ZOO is to optimize a smoothed surrogate of the original objective, typically constructed via randomized finite differences. Let $\rmL \in \sR^{d \times d}$ denote a smoothing matrix (e.g., $\rmL = \mu \rmI$ in standard scalar smoothing) and $\rvu \sim \gN(\vzero, \rmI_d)$ be a random perturbation vector. The smoothed objective $F_{\rmL}(\vtheta)$ is defined as:
$$
F_{\rmL}(\vtheta) \triangleq \E_{\rvu}\left[F(\vtheta + \rmL\rvu)\right] = \E_{\rvu, \xi}\left[f(\vtheta + \rmL\rvu; \xi)\right].
$$
To optimize $F_{\rmL}$, a widely used gradient estimator based on randomized finite differences is given by:
$$
\hat{\nabla} F_{\rmL}(\vtheta) \triangleq \frac{1}{K} \sum_{k=1}^K \left[ f(\vtheta + \rmL\rvu_{t,k}; \xi) - f(\vtheta; \xi) \right] \rmL^{-\top}\rvu_{t,k}.
$$
This estimator is an unbiased approximation of $\nabla F_{\rmL}(\vtheta)$ under Gaussian smoothing, i.e., $\E_{\rvu, \xi}\left[\hat{\nabla} F_{\rmL}(\vtheta)\right] = \nabla_{\vtheta} F_{\rmL}(\vtheta)$.

**The Policy Optimization Perspective.** While historically rooted in finite difference approximations, ZOO can be rigorously reinterpreted as a specific instance of *single-step Policy Optimization (PO)*. Consider a single-step RL agent defined by a stochastic policy $\pi_{\vtheta}(\rvx)$ that samples an action $\rvx$ and receives a reward $R(\rvx)$. Let the policy be parameterized as a multivariate Gaussian distribution $\pi_{\vtheta}(\rvx) = \gN(\rvx \mid \vtheta, \vSigma)$, where $\vSigma = \rmL\rmL^\top$. By defining the reward as the negative loss, $R(\rvx) = -f(\rvx; \xi)$, the standard objective in policy optimization, maximizing expected reward, becomes equivalent to minimizing the smoothed ZOO objective:
$$
J(\vtheta) \triangleq \E_{\rvx \sim \pi_{\vtheta}} [-R(\rvx)] = \E_{\rvu} [F(\vtheta + \rmL\rvu)] \equiv F_{\rmL}(\vtheta).
$$
Applying the REINFORCE w/ baseline estimator to the objective $J(\vtheta)$ yields:
$$
\nabla J(\vtheta) = \E_{\rvx \sim \pi_{\vtheta}} \left[ \nabla_{\vtheta} \log \pi_{\vtheta}(\rvx) (R(\rvx) - b) \right].
$$
For the Gaussian policy $\pi_{\vtheta}(\rvx) = \gN(\vtheta, \rmL\rmL^\top)$, the score function is $\nabla_{\vtheta} \log \pi_{\vtheta}(\rvx) = \rmL^{-\top}\rmL^{-1}(\rvx - \vtheta) = \rmL^{-\top}\rvu$. Let $b=f(\vtheta;\xi)$, substituting this into the policy gradient formulation recovers the ZOO estimator exactly:
$$
\nabla J(\vtheta) = \E_{\rvu, \xi} \left[ (f(\vtheta + \rmL\rvu; \xi) - b) \rmL^{-\top}\rvu \right] \equiv \nabla F_{\rmL}(\vtheta).
$$
This equivalence reveals that the finite-difference term $f(\vtheta; \xi)$ commonly subtracted in ZOO is not merely a Taylor expansion artifact, but constitutes a *variance-reducing baseline* in the context of the REINFORCE algorithm.

**Motivation for Adaptive Smoothing.** Standard ZOO methods typically rely on a fixed, scalar smoothing parameter (i.e., $\rmL = \mu \rmI$), which imposes two critical limitations.
**First**, it enforces *isotropic* smoothing, utilizing a search distribution that explores all dimensions equally. This is notoriously inefficient for ill-conditioned landscapes where sensitivity varies significantly across directions.
**Second**, it maintains a *fixed smoothing scale*, failing to adapt the exploration radius to the optimization stage: large perturbations are needed for initial exploration, while small perturbations are crucial for final convergence.
Crucially, the PO equivalence established above identifies $\rmL$ as the generator of the *exploration covariance* $\vSigma = \rmL\rmL^\top$. Just as modern RL algorithms optimize the full covariance to adapt to the reward landscape, ZOO can fundamentally benefit from learning a general matrix $\rmL$. This insight motivates our proposed *adaptive smoothing* framework, which leverages this connection to dynamically optimize $\rmL$, simultaneously enabling *anisotropic sampling* to align with the landscape geometry and *varying smoothing* to adjust the exploration scale.

## Zeroth-Order Optimization w/ Adaptive Smoothing

Building upon the theoretical foundations introduced in \cref{sec:preliminaries}, we now derive our adaptive smoothing framework. We first demonstrate that the update rule of standard ZOO is implicitly solving a *constrained* variational problem. By relaxing these constraints, we derive a closed-form mechanism for simultaneously optimizing the search mean and the covariance matrix, thereby enabling adaptive exploration.

### A Variational Perspective on Standard ZOO

In \cref{sec:preliminaries}, we established the equivalence between ZOO and single-step PO, identifying the standard smoothed objective $F_{\rmL}(\vtheta)$ as the expected loss under a Gaussian policy $\pi_{\vtheta}$ with fixed covariance $\boldsymbol{\Sigma} = \rmL\rmL^\top$. While this equivalence allows us to view ZOO as policy optimization, directly minimizing the policy objective $\E_{\pi}[F]$ with stochastic gradients estimates is notoriously unstable in practice. A single large update based on a noisy gradient estimate can push the parameters into a region of high loss. To address this, modern Reinforcement Learning (e.g., TRPO, PPO) and LLM alignment (e.g., RLHF) rely on KL-regularized optimization defined as below. By enforcing a proximity constraint between the new and old policies, these methods ensure that updates remain within a neighborhood where the local gradient information is valid.
$$
\min_{\pi \in \Pi} \; \mathcal{J}(\pi) \triangleq \underbrace{\E_{\rvx \sim \pi}[F(\rvx)]}_{\text{Optimization Target}} + \underbrace{\beta D_{\mathrm{KL}}(\pi \,\|\, \pi_{\text{\normalfont ref}})}_{\text{Stability Constraint}},
$$
where $\beta > 0$ is an inverse temperature parameter that controls the strength of the KL-divergence penalty.

Adopting this rigorous lens, we posit that standard ZOO update at every step $t+1$ are implicitly solving the KL-regularized variational problem above in Prop. \ref{prop:standard-zoo-proximal} below.

> **Proposition 1.**
> Let $\Pi_{\text{\normalfont fixed}} = \{ \gN(\vtheta, \boldsymbol{\Sigma}) \mid \boldsymbol{\Sigma} = \rmL\rmL^\top \}$ denote the family of Gaussian policies constrained to a fixed geometry. The optimization of the functional $\mathcal{J}(\pi)$ in Eq (8) over $\Pi_{\text{\normalfont fixed}}$ with $\pi_{\text{\normalfont ref}} = \gN(\vtheta_t, \rmL\rmL^\top)$ is equivalent to minimizing the regularized parametric objective:
> $$
> \widetilde{F}_{\rmL}(\vtheta) \triangleq F_{\rmL}(\vtheta) + \frac{\beta}{2}\|\rmL^{-\top}(\vtheta - \vtheta_t)\|^2.
> $$
> So, the standard ZO-SGD on $F_{\rmL}$ at step $t+1$ corresponds exactly to finding a minimum (global for convex or local for non-convex) on $\widetilde{F}_{\rmL}$:
> $$
> \begin{aligned}
> \vtheta_{t+1} = \vtheta_t - \frac{1}{\beta}\rmL\rmL^\top \hat{\nabla} F_{\rmL}(\vtheta_t),
> \end{aligned}
> $$
> where $\hat{\nabla} F_{\rmL}(\vtheta_t)$ is the standard ZO gradient estimator.

*Proof (Sketch).* For Gaussian policies sharing identical covariance $\boldsymbol{\Sigma}$, the KL divergence simplifies to the Mahalanobis distance: $D_{\mathrm{KL}}(\pi_{\vtheta} \| \pi_{\vtheta_t}) = \frac{1}{2}(\vtheta - \vtheta_t)^\top \boldsymbol{\Sigma}^{-1} (\vtheta - \vtheta_t)$. Minimizing the first-order approximation of $F_{\rmL}$ subject to this quadratic penalty yields the closed-form update $\vtheta_{t+1} = \vtheta_t - \frac{1}{\beta}\boldsymbol{\Sigma}\nabla F_{\rmL}(\vtheta_t)$. Substituting $\boldsymbol{\Sigma}=\rmL\rmL^\top$ recovers the standard preconditioned ZOO update, where the learning rate is identified as $1/\beta$.

**Implication.**
While Prop. \ref{prop:standard-zoo-proximal} legitimizes standard ZOO as a stable policy update, it exposes a fundamental source of suboptimality, i.e., the *fixed smoothing restriction*. That is standard ZOO forces the smoothing matrix $\boldsymbol{\Sigma}$ to remain isotropic ($\rmL\rmL^\top = \mu^2 \rmI$), preventing the search distribution from adapting to the underlying function curvature.
This insight motivates our proposed approach: we move beyond the *fixed smoothing* update to an *adaptive smoothing* one that leverages the full variational objective in Eq (8).

### Theoretically Optimal Gaussian Smoothing

To overcome the aforementioned limitations of the fixed smoothing step, we lift the restrictions on the policy space and optimize Eq (8) directly. Specifically, we propose to seek the global minimizer of the exact functional $\mathcal{J}(\pi)$ over the space of probability measures, and then project this optimal solution back onto the Gaussian manifold. This approach is superior because it captures the global structure of the local landscape (via energy-based weighting) rather than relying solely on local gradient approximations.

We proceed in the following two steps: (1) deriving the optimal non-parametric policy in \cref{thm:kl-optimal-policy}, and (2) projecting it to the optimal Gaussian distribution in \cref{prop:adaptive-updates} to derive closed-form updates for both mean and covariance.

> **Theorem 2.**
> The unique global minimizer $\pi^\star$ of the unrestricted functional $\mathcal{J}(\pi)$ in Eq (8) is the energy-based Boltzmann distribution:
> $$
> \pi^{\star}(\rvx) = \frac{1}{Z}\pi_{\text{\normalfont ref}}(\rvx) \exp\left(-\frac{F(\rvx)}{\beta}\right)
> $$
> where $Z = \E_{\pi_{\text{\normalfont ref}}}\left[\exp\left(-\frac{F(\rvx)}{\beta}\right)\right]$ is the partition function.

*Proof.* This result follows from the functional derivative of the Lagrangian $\mathcal{L}(\pi) = \mathcal{J}(\pi) + \lambda(\int \pi(\rvx)d\rvx - 1)$. Setting the functional gradient $\delta \mathcal{L} / \delta \pi = 0$ yields the condition $F(\rvx) + \beta(1 + \log \frac{\pi}{\pi_{\vtheta_t}}) + \lambda = 0$, which implies the Boltzmann form $\pi \propto \pi_{\vtheta_t} \exp(-F/\beta)$.

The optimal policy $\pi^\star$ captures the ideal search distribution: it reweights the prior $\pi_\text{\normalfont ref}$ to concentrate probability mass in regions of low loss $F(\rvx)$, naturally embodying the landscape geometry. However, $\pi^\star$ is generally non-Gaussian and intractable to sample from directly.

To derive a practical algorithm, we project $\pi^\star$ onto the Gaussian family $\Pi = \{ \gN(\vtheta, \boldsymbol{\Sigma}) \}$ by finding $\pi \in \Pi$ that minimizes the reverse KL divergence $D_{\mathrm{KL}}(\pi^\star \| \pi)$, thereby preserving the first and second-order information of the optimal policy.
$$
\widehat{\pi}^* = \argmin_{\pi \in \Pi} D_{\mathrm{KL}}(\pi^\star \| \pi)
$$

> **Theorem 3.**
> The optima $\widehat{\pi}^* = \gN(\vtheta^*, \boldsymbol{\Sigma}^*)$ achieves identical first and second moments of $\pi^\star$. This yields the following closed-form updates:
> $$
> \begin{align}
> \vtheta^* &= \E_{\rvx \sim \pi_{\text{\normalfont ref}}} \left[ w(\rvx) \rvx \right], \\
> \boldsymbol{\Sigma}^* &= \E_{\rvx \sim \pi_{\text{\normalfont ref}}} \left[ w(\rvx) (\rvx - \vtheta^*)(\rvx - \vtheta^*)^\top \right],
> \end{align}
> $$
> where $w(\rvx) = \frac{1}{Z}\exp\left(-\frac{F(\rvx)}{\beta}\right)$ are the normalized importance weights.

**Theoretical Implications.** Unlike standard ZOO, which relies on local linearizations to estimate descent directions, our variational update leverages the global energy landscape of the objective function. By matching the moments of the Boltzmann distribution, the update implicitly learns the local curvature, causing the search covariance to expand in flat plateaus for acceleration and contract near sharp minima for stability. This acts as a learnable, Hessian-aware preconditioner that dynamically adapts both the orientation and scale of exploration without manual tuning, offering a fundamental advantage over fixed-geometry proximal steps.

**From Theory to Practice.** While Theorem~\ref{prop:adaptive-updates} characterizes the optimal updates at the population level $\E_{\pi_t}$, practical implementation faces the constraint of finite function queries and the prohibitive cost of storing dense covariance matrices in high dimensions. In the next subsection, we introduce a scalable algorithm that approximates these expectations efficiently using a memory-efficient low-rank decomposition.

### Practical Implementation

In the zeroth-order setting, the analytical expectations in \cref{prop:adaptive-updates} are intractable. We approximate them using the $K$ function queries collected at iteration $t$. Let $\{\rvx_{t,k}\}_{k=1}^K$ be the samples drawn from the current policy $\pi_t = \gN(\vtheta_t,\vSigma_t)$ with $\vSigma_t = \rmL\rmL^{\top}$ and $\{\rvu_{t,k}\}_{k=1}^K$ be the corresponding random vectors drawn from $\gN(\vzero,\rmI)$ according to reparameterization trick $\rvx_{t,k} = \vtheta_t + \rmL_t\rvu_{t,k}$. Define
$$
w_{t,k} = \frac{\exp\left(-f(\rvx_{t,k}; \xi_k)/\beta\right)}{\sum_{j=1}^K \exp\left(-f(\rvx_{t,j}; \xi_j)/\beta\right)}, \quad \bar{\rvu}_t \triangleq \sum_{k=1}^K w_{t,k} \rvu_{t,k},
$$
the parameters for the next iteration are estimated via self-normalized importance sampling:
$$
\vtheta_{t+1} = \sum_{k=1}^K w_{t,k} \rvx_{t,k} = \vtheta_t + \rmL_{t} \bar{\rvu}_t
$$

$$
\begin{aligned}
\boldsymbol{\Sigma}_{t+1} &= \sum_{k=1}^K w_{t,k} (\rvx_{t,k} - \vtheta_{t+1})(\rvx_{t,k} - \vtheta_{t+1})^\top \\
&= \sum_{k=1}^K w_{t,k} \left[ \rmL_t (\rvu_{t,k} - \bar{\rvu}_t) \right] \left[ \rmL_t (\rvu_{t,k} - \bar{\rvu}_t) \right]^\top \\
&= \rmL_t \left( \sum_{k=1}^K w_{t,k} \rvu_{t,k} \rvu_{t,k}^\top - \bar{\rvu}_t \bar{\rvu}_t^\top \right) \rmL_t^\top.
\end{aligned}
$$

define $\rmR_t = [\sqrt{w_{t,1}}(\rvu_{t,1} - \bar{\rvu}_t), \dots, \sqrt{w_{t,K}}(\rvu_{t,K} - \bar{\rvu}_t)]$
$$
\rmL_{t+1} = \rmL_{t}\rmR_t
$$

This formulation allows us to update the search geometry entirely from the query history without requiring additional function evaluations or gradient approximations.

The update for the smoothing matrix $\rmL_{t+1}$ can be derived efficiently by factorizing the covariance in the latent space. Let $\mathbf{C}_t^{\rvu} \in \sR^{K \times K}$ denote the weighted sample covariance of the latent perturbations:
$$
\mathbf{C}_t^{\rvu} \triangleq \sum_{k=1}^K w_{t,k} (\rvu_{t,k} - \bar{\rvu}_t)(\rvu_{t,k} - \bar{\rvu}_t)^\top.
$$
Observing that $\boldsymbol{\Sigma}_{t+1} = \rmL_t \mathbf{C}_t^{\rvu} \rmL_t^\top$, we can update the smoothing matrix by propagating the square root of the latent covariance:
$$
\rmL_{t+1} = \rmL_t \mathbf{R}_t, \quad \text{where } \mathbf{R}_t \mathbf{R}_t^\top = \mathbf{C}_t^{\rvu}.
$$
In practice, we compute $\mathbf{R}_t$ directly by stacking the weighted latent residuals. Let $\mathbf{R}_t \in \sR^{K \times K}$ be the matrix with columns $\mathbf{r}_k = \sqrt{w_{t,k}}(\rvu_{t,k} - \bar{\rvu}_t)$. The update simplifies to a matrix multiplication $\rmL_{t+1} \leftarrow \rmL_t \mathbf{R}_t$, which entails a computational complexity of $\mathcal{O}(dK^2)$, linear in the parameter dimension $d$.