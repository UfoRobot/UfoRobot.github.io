---
layout: post
title: The .flac format: linear predictive coding and rice codes
mathjax: true
---
**FLAC** stands for **F**ree **L**ossless **A**udio **C**odec. It's the standard open source codec used nowadays for lossless audio compression. In this post I would like to take the .flac format as an example and focus on two core step of the process of lossless audio compression: linear predictive coding and entropy coding with Rice codes.

## TL;TR
Most lossless audio compression codecs work very similarly: first the audio stream is split into blocks, and then each one is compressed. Compression is achieved identifying and modelling structure in the signal: any explicit structure and repeating pattern is redundant by definition and can be instead represented more efficiently using a mathematical model and its parameters to approximate it. Usually a very simple model is used, from constant values to model silence to the more flexible linear predictive coding (a linear autoregressive model). In order to recreate the signal exactly, the approximation residuals are also saved but using a coding scheme optimised for their distribution. Linear predictive coding leads to residual Laplace-distributed and hence usually the Rice coding scheme is used. 


Linear predictive coding is a modelling technique equivalent to an AR(p) auto regressive model: the signal \(X_t\) at time \(t\) is modelled as a linear combination of its previous \(p\) values.\\

More formally, let \(\{X_t\}\) be a wide sense stationary (WSS) stochastic process with zero mean and with the \(AR(p)\) property, which is:

\begin{equation}\label{eqn:wss_1}
E[X_t] = \mu = 0  & \quad \forall t
\end{equation}
\begin{equation}\label{eqn:wss_2}
Cov[X_{t}, X_{t + \tau}] = Cov[X_{t+h}, X_{t +h + \tau}]  & \quad \forall t \forall h   
\end{equation}

From \ref{eqn:wss_1} and \ref{eqn:wss_2} it follows that the covariance function only depends on the lag \(\tau\) and can be expressed in terms of the autocorrelation function \(R(\tau)\):

\begin{equation}\label{eqn:autocorr}
Cov[X_{t}, X_{t + \tau}] = E[(X_t - \mu)(X_{t + \tau} - \mu)] = E[X_tX_{t+\tau}] = R(\tau)
\end{equation}
where \(R(\tau) = R(-\tau)\) is symmetric.\\
   

Assume that we wanted to approximate each value \(X_t\) as a linear combination of its previous p values:
\[ \hat{X_t} = \sum_{k = 1}^{p}\alpha_k X_{t-k} \]
now let \(\boldsymbol{\boldsymbol{\alpha}} =  \begin{bmatrix} \alpha_1 & \hdots & \alpha_p \end{bmatrix}^\top \), the approximation error \(\epsilon_t\) at time t is the random variable:

\[ \epsilon_t    =  X_t - \hat{X_t} 
            =  X_t - \mathbf{\boldsymbol{\alpha}}^\top \begin{bmatrix} X_{t-1} \\ \vdots \\ X_{t-p} \end{bmatrix} 
            =  X_t - \mathbf{\boldsymbol{\alpha}}^\top \mathbf{X_{t-1:t-p}} \]
We can then chose \(\boldsymbol{\alpha}\) so that it minimises the expected squared estimation error.\\


Lemma: Orthogonality principle:
The estimate \(\hat{X_t} \) minimises the expected squared error if and only if the estimation error \(\epsilon_t\) is orthogonal to the p random variables \(X_{t-1}, ..., X_{t-p}\), ie if and only if:

\[E[\epsilon_t X_{t-i}] = 0 \quad \forall 1 \leq i \leq p\]

We can use the orthogonality principle to derive the Yule-Walker equations:
\begin{equation}\label{eqn:yw}
\begin{split}
\mathbf{0} = & E[\epsilon_t \mathbf{X_{t-1:t-p}}] \\
           = & E[(X_t - \mathbf{X_{t-1:t-p}}^\top\alpha)\mathbf{X_{t-1:t-p}}] \\
           = & E[X_t \mathbf{X_{t-1:t-p}} - \mathbf{X_{t-1:t-p}}^\top\boldsymbol{\alpha}\mathbf{X_{t-1:t-p}}] \\
            = & E[X_t \mathbf{X_{t-1:t-p}} - trace(\mathbf{X_{t-1:t-p}}^\top\boldsymbol{\alpha}\mathbf{X_{t-1:t-p}})] \\
            = & E[X_t \mathbf{X_{t-1:t-p}} - trace(\mathbf{X_{t-1:t-p}}\mathbf{X_{t-1:t-p}}^\top\boldsymbol{\alpha})] \\
            = & E[X_t \mathbf{X_{t-1:t-p}} - \mathbf{X_{t-1:t-p}}\mathbf{X_{t-1:t-p}}^\top\boldsymbol{\alpha}] \\
            = & E[X_t \mathbf{X_{t-1:t-p}}] - E[ \mathbf{X_{t-1:t-p}}\mathbf{X_{t-1:t-p}}^\top]\boldsymbol{\alpha} \\
\end{split}
\end{equation}

Using \ref{eqn:autocorr} we can rewrite \( E[ \mathbf{X_{t-1:t-p}}\mathbf{X_{t-1:t-p}}]^\top = \mathbf{R}\)  as:

\[\mathbf{R} = 
\begin{bmatrix}
R(0) & R(1) & R(2) & \hdots & R(p-1) \\
R(1) & R(0) & R(1) & \hdots & R(p-2) \\
R(2) & R(1) & R(0) & \hdots & R(p-3) \\
\vdots  & \vdots   & \vdots &  & \vdots \\
R(p-1) & R(p-2) & R(p-3) & \hdots & R(0) \\ 
\end{bmatrix}
\]

and by letting \(\mathbf{r} = \begin{bmatrix} R(1) & \hdots & R(p) \end{bmatrix}^\top\) we can rewrite \ref{eqn:yw} as:

\begin{equation}\label{eqn:yh_2}
\mathbf{R}\boldsymbol{\alpha} = \mathbf{r}   
\end{equation}


which can be solved for \(\boldsymbol{\alpha}\) given estimates of \(\mathbf{R}\) and \(\mathbf{r}\)

Note that the matrix \(\mathbf{R}\) is a \textit{toeplitz} matrix. A toeplitz matrix has the property that each descending diagonal from left to right is constant and there exist an algorithm, the Levinson–Durbin recursion, to solve the system in \(O(p^2)\) instead of simply inverting \(\mathbf{R}\), which would have a cost \(O(p^3)\). The overall computational cost is thus \(O(np + p^2)\) where the first term comes from the cost of computing the sample estimate of the auto correlation matrix.





