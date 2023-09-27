**QPLayer** enables to use a QP as a layer within standard learning architectures. More precisely, QPLayer differentiates over $\theta$ the primal and dual solutions of QP of the form

$$
\begin{align}
\min_{x} &  ~\frac{1}{2}x^{T}H(\theta)x+g(\theta)^{T}x \\
\text{s.t.} & ~A(\theta) x = b(\theta) \\
& ~l(\theta) \leq C(\theta) x \leq u(\theta)
\end{align}
$$

where $x \in \mathbb{R}^n$ is the optimization variable. The objective function is defined by a positive semidefinite matrix $H(\theta) \in \mathcal{S}^n_+$ and a vector $g(\theta) \in \mathbb{R}^n$. The linear constraints are defined by the equality-contraint matrix $A(\theta) \in \mathbb{R}^{n_\text{eq} \times n}$ and the inequality-constraint matrix $C(\theta) \in \mathbb{R}^{n_\text{in} \times n}$ and the vectors $b \in \mathbb{R}^{n_\text{eq}}$, $l(\theta) \in \mathbb{R}^{n_\text{in}}$ and $u(\theta) \in \mathbb{R}^{n_\text{in}}$ so that $b_i \in \mathbb{R},~ \forall i = 1,...,n_\text{eq}$ and $l_i \in \mathbb{R} \cup \{ -\infty \}$ and $u_i \in \mathbb{R} \cup \{ +\infty \}, ~\forall i = 1,...,n_\text{in}$.

We provide in the file qplayer_sudoku.py an example which enables training LP layer in two different settings: (i) either we learn only the equality constraint matrix $A$, or (ii) we learn on the same time $A$ and $b$, such that $b$ is structurally in the range space of $A$. The procedure (i) is harder since a priori the fixed right hand side does not ensure the QP to be feasible. Yet, this learning procedure is more structured, and for some problem can produce better prediction quicker (i.e., in fewer epochs).