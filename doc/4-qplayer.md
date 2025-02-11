## QPLayer

**QPLayer** enables to use a QP as a layer within standard learning architectures. More precisely, QPLayer differentiates over \f$\theta\f$ the primal and dual solutions of QP of the form

$$
\begin{align}
\min_{x} &  ~\frac{1}{2}x^{T}H(\theta)x+g(\theta)^{T}x \\\
\text{s.t.} & ~A(\theta) x = b(\theta) \\\
& ~l(\theta) \leq C(\theta) x \leq u(\theta)
\end{align}
$$

where \f$x \in \mathbb{R}^n\f$ is the optimization variable. The objective function is defined by a positive semidefinite matrix \f$H(\theta) \in \mathcal{S}^n_+\f$ and a vector \f$g(\theta) \in \mathbb{R}^n\f$. The linear constraints are defined by the equality-contraint matrix \f$A(\theta) \in \mathbb{R}^{n_\text{eq} \times n}\f$ and the inequality-constraint matrix \f$C(\theta) \in \mathbb{R}^{n_\text{in} \times n}\f$ and the vectors \f$b \in \mathbb{R}^{n_\text{eq}}\f$, \f$l(\theta) \in \mathbb{R}^{n_\text{in}}\f$ and \f$u(\theta) \in \mathbb{R}^{n_\text{in}}\f$ so that \f$b_i \in \mathbb{R},~ \forall i = 1,...,n_\text{eq}\f$ and \f$l_i \in \mathbb{R} \cup \{ -\infty \}\f$ and \f$u_i \in \mathbb{R} \cup \{ +\infty \}, ~\forall i = 1,...,n_\text{in}\f$.

We provide in the file `qplayer_sudoku.py` an example which enables training LP layer in two different settings: (i) either we learn only the equality constraint matrix \f$A\f$, or (ii) we learn on the same time \f$A\f$ and \f$b\f$, such that \f$b\f$ is structurally in the range space of \f$A\f$. The procedure (i) is harder since a priori the fixed right hand side does not ensure the QP to be feasible. Yet, this learning procedure is more structured, and for some problem can produce better prediction quicker (i.e., in fewer epochs).

The differentiable QP layer is implemented in \ref proxsuite.torch.qplayer.QPFunction.

\section QPLayerCite How to cite QPLayer ?

If you are using QPLayer for your work, we encourage you to cite the related paper with the following format.
\code
@unpublished{bambade:hal-04133055,
  TITLE = {{QPLayer: efficient differentiation of convex quadratic optimization}},
  AUTHOR = {Bambade, Antoine and Schramm, Fabian and Taylor, Adrien and Carpentier, Justin},
  URL = {https://inria.hal.science/hal-04133055},
  NOTE = {working paper or preprint},
  YEAR = {2023},
  MONTH = Jun,
  KEYWORDS = {Machine Learning ; Optimization ; Differentiable Optimization ; Optimization layers},
  PDF = {https://inria.hal.science/hal-04133055/file/QPLayer_Preprint.pdf},
  HAL_ID = {hal-04133055},
  HAL_VERSION = {v1},
}
\endcode

The paper is publicly available in HAL ([ref 04133055](https://inria.hal.science/hal-04133055/file/QPLayer_Preprint.pdf)).
