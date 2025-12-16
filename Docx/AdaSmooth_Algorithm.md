
\begin{algorithm}[tb]
   \caption{\AlgName{}: Anisotropic Zeroth-Order Optimization}
   \label{alg:adasmooth-lowrank}
\begin{algorithmic}[1]
   \STATE {\bfseries Input:} Initial $\vtheta_0$, smoothing matrix $\rmL_0 \in \R^{d \times K}$, temperature schedule $\{\beta_t\}$, batch size $K$.
   
   \FOR{$t = 0$ {\bfseries to} $T-1$}
      \STATE \emph{// 1. Anisotropic Sampling}
      \FOR{$k = 1$ {\bfseries to} $K$}
         \STATE Sample latent vector: $\rvu_{t,k} \sim \gN(\vzero, \rmI_K)$.
         \STATE Construct candidate: $\rvx_k = \vtheta_t + \rmL_t \rvu_{t,k}$.
         \STATE Query oracle: $y_k = f(\rvx_k; \xi_k)$.
      \ENDFOR
      
      \STATE \emph{// 2. Compute Importance Weights}
      \STATE Compute raw weights: $v_k = \exp(-y_k/\beta_t)$.
      \STATE Normalize: $w_{t,k} = v_k / \sum_{j=1}^K v_j$.
      
      \STATE \emph{// 3. Update Parameters}
      \STATE Update mean: $\vtheta_{t+1} = \sum_{k=1}^K w_{t,k} \rvx_k$.
      \STATE Compute weighted residuals: $\vc_k = \sqrt{w_{t,k}} (\rvx_k - \vtheta_{t+1})$.
      \STATE Update smoothing matrix (Rank-$K$):
      \STATE $\rmL_{t+1} \leftarrow [\vc_1, \vc_2, \dots, \vc_K] \in \R^{d \times K}$.
      
   \ENDFOR
   \STATE {\bfseries Return} $\vtheta_T$.
\end{algorithmic}
\end{algorithm}