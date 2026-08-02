# GNN Basics — Notes from `03-GNN.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Source:** Lecture 3 slides (`03-GNN.pdf`)

**Related readings (from the deck):**
- Graph representation learning: methods and applications
- Inductive Representation Learning for Large Graphs (GraphSAGE)
- Lecture 2: PageRank / Personalized PageRank (PPR)

---

## 0. Big picture

**Goal of node embeddings:** map each node $v$ to a $d$-dimensional vector so that *similar* nodes in the graph land close in embedding space.

$$
z_v = f(v;\text{graph})
$$

**Shallow encoders** (e.g. embedding lookup table) learn one vector per node.  
**Deep graph encoders = Graph Neural Networks (GNNs):** multi-layer nonlinear transforms that use the *graph structure* (and node features) to produce embeddings.

GNNs can produce:
- **node** embeddings
- **subgraph** embeddings
- **whole-graph** embeddings

**Lecture structure:**
1. Basics of deep learning (needed to train GNNs)
2. Deep learning for graphs (message-passing / neighborhood aggregation)

---

## 1. Basics of deep learning

### 1.1 ML as optimization

**Supervised learning:** given input $x$, predict label $y$.

Inputs can be:
- vectors
- sequences (text)
- matrices (images)
- **graphs** (with optional node/edge features)

Formulate as:

$$
\min_{\Theta}\; \mathcal{L}\big(y,\, f(x)\big)
$$

| Symbol | Meaning |
|--------|---------|
| $\Theta$ | parameters we optimize (scalars, vectors, matrices, ...) |
| $f$ | model (linear layer, MLP, GNN, ...) |
| $\mathcal{L}$ | loss |

**Common losses:** L2, L1, Huber, hinge / max-margin, cross-entropy.  
(PyTorch: `torch.nn` loss functions.)

In shallow encoders, $\Theta$ can literally be the embedding table $Z$.

---

### 1.2 Cross-entropy + softmax (classification)

Model outputs logits $f(x)$, e.g. $[0.1, 0.1, 0.6, 0.2, 0]$.  
Label $y$ is **one-hot**, e.g. $[0,0,1,0,0]$ → class 3.

**Softmax** (differentiable "soft max"):

$$
\text{Softmax}(f(x))_i = \frac{e^{f(x)_i}}{\sum_{j=1}^{C} e^{f(x)_j}}
$$

Turns logits into a probability distribution over $C$ classes.

**Cross-entropy:**

$$
\text{CE}(y, f(x)) = -\sum_{i=1}^{C} y_i \log f(x)_i
$$

- With one-hot $y$, only **one term** is nonzero.
- Intuition: lower CE ⇔ predicted distribution closer to the one-hot target.

**Total training loss:**

$$
\mathcal{L} = \sum_{(x,y)\in\mathcal{T}} \text{CE}(y, f(x))
$$

---

### 1.3 Why gradients?

We need scalable optimization → **gradient-based** methods.  
So $\mathcal{L}$ should be **differentiable**.

Non-gradient alternatives exist (Bayesian opt, GPs, simulated annealing, evolutionary methods) but are less common at deep-learning scale.

For non-differentiable pieces, people use tricks like:
- straight-through estimator / Gumbel-Softmax
- REINFORCE / RL-style estimators

**Gradient of the loss:**

$$
\nabla_{\Theta}\mathcal{L} = \Big(\frac{\partial\mathcal{L}}{\partial\Theta_1},\, \frac{\partial\mathcal{L}}{\partial\Theta_2},\, \ldots\Big)
$$

Points in the direction of **steepest increase**. Training steps go the **opposite** way.

---

### 1.4 Gradient descent & SGD

**Gradient descent update:**

$$
\Theta \leftarrow \Theta - \eta\, \nabla_{\Theta}\mathcal{L}
$$

| Term | Meaning |
|------|---------|
| **Iteration** | one gradient step |
| **Learning rate** $\eta$ | step size (hyperparameter; often scheduled) |
| Ideal stop | gradient ≈ 0 |
| Practical stop | validation performance stops improving (early stopping) |

**Problem with full GD:** exact $\nabla\mathcal{L}$ needs the **entire dataset** every step → too expensive (datasets can be huge).

**Stochastic Gradient Descent (SGD):**
- each step, sample a **minibatch** $\mathcal{B}$
- approximate the gradient on that batch only

**Minibatch SGD vocabulary:**

| Term | Meaning |
|------|---------|
| **Batch size** | # examples in a minibatch (e.g. # nodes for node classification) |
| **Iteration** | one SGD step on one minibatch |
| **Epoch** | one full pass over the dataset $\approx$ (# examples) / (batch size) iterations |

SGD is an **unbiased** estimator of the full gradient, but convergence rate is not free — usually needs LR tuning.

**Better optimizers (common practice):** Adam, Adagrad, Adadelta, RMSprop, ...  
**Adam** is often a solid default.  
Also: LR annealing / schedulers (e.g. tanh schedule) help **convergence + generalization**.

---

### 1.5 Neural net functions & backprop

Start simple — linear model:

$$
f(x) = W\cdot x,\quad \Theta=\{W\}
$$

- scalar output → $W$ is a vector  
- vector output → $W$ is a matrix  

**Higher-order tensors** (derivatives of matrix→matrix, etc.) get messy; GPUs are optimized for **matrix** ops.

**Deeper linear stack still linear:**

$$
f(x) = W_2 W_1 x
$$

Composing linear maps is still linear. Need **nonlinearity**.

#### Backpropagation (chain rule)

For $f(x) = a = W_2 W_1 x$:
- hidden: $z = W_1 x$
- output: $a = W_2 z$

**Forward:** $x \to z \to a \to \mathcal{L}$  
**Backward:** start from loss, chain-rule gradients into each $\Theta$.

$$
\frac{\partial\mathcal{L}}{\partial W_2}
= \frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial W_2}
,\qquad
\frac{\partial\mathcal{L}}{\partial W_1}
= \frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial W_1}
$$

**Concrete toy numbers (from slides):**  
$x^\top=[0.8, 1.1]$, $y=1$,  
$W_1=\begin{bmatrix}0.1&0.2\\0.3&0.4\end{bmatrix}$, $W_2=[0.5,\,0.6]$  
→ $a = W_2 W_1 x = 0.558$, $\mathcal{L}=(y-a)^2=0.1954$, then backprop for $\partial\mathcal{L}/\partial W$.

---

### 1.6 Nonlinearity & MLP

**ReLU:** $\operatorname{ReLU}(x)=\max(x,0)$  
**Sigmoid:** $\sigma(x)=\dfrac{1}{1+e^{-x}}$

**MLP layer:**

$$
x^{(l+1)} = \sigma\big(W^{(l)} x^{(l)} + b^{(l)}\big)
$$

Each layer = **linear transform + nonlinearity**.

**Universal approximation:** sufficiently wide MLPs can approximate many continuous functions on compact sets — but caveats matter:
- embedding / width dimension
- generalization
- convergence / optimization
- invariances you *want* the model to respect (critical for graphs)

---

### 1.7 DL summary (pre-GNN)

1. Objective: $\min_\Theta \mathcal{L}(y, f(x))$
2. Sample a minibatch
3. **Forward** → compute loss
4. **Backward** → $\nabla_\Theta\mathcal{L}$ via chain rule
5. **SGD / Adam** update $\Theta$ for many iterations

$f$ can later be a **GNN**.

---

## 2. Deep learning for graphs

### 2.1 Setup (recap)

Graph $G$:
- $V$: vertices
- $A$: adjacency matrix (often binary in the lecture)
- $X \in \mathbb{R}^{d \times |V|}$: node feature matrix
- $v \in V$, $N(v)$: neighbors of $v$

**Example node features:**
- social nets: profile, image features
- bio nets: gene expression, functional annotations

**If no features exist:**
- one-hot indicator per node, or
- constant vector $[1,1,\ldots,1]$

---

### 2.2 Naive approach (and why it fails)

Idea:
1. concatenate $[A, X]$
2. feed into a big fully connected net

**Problems:**
- **Huge** parameter count — $O(|V|)$ parameters (scales with graph size)
- **Not inductive** — hard / impossible to apply to graphs of different size
- **Node-order sensitive** — permuting rows/columns of $A$ changes the input even if the graph is the same

We need models that respect **permutation / graph structure**.

---

### 2.3 Motivation from CNNs

On images, CNNs:
- use **local** neighborhoods (e.g. 3×3 filters)
- **share** weights across locations
- leverage pixel features

**Want the same idea on graphs:**
- generalize convolution beyond grids
- use node attributes
- but real graphs have **no fixed grid / sliding window**
- graphs are **permutation invariant**

---

### 2.4 Core GNN idea: neighborhood = computation graph

**Key idea:** a node's **local neighborhood** defines a **computation graph**.

At each layer, nodes:
1. gather "messages" from neighbors
2. transform them with neural nets
3. aggregate into an updated embedding

Intuition from CNN → graph:
- transform neighbor messages: $W_i h_i$
- combine (e.g. sum): $\sum_i W_i h_i$

---

### 2.5 Stacking layers (depth = hop distance)

- **Layer-0** embedding of $u$: its raw features $x_u$
- **Layer-$k$** embedding of $u$: information from nodes up to **$k$ hops** away
- Model depth $L$ can be arbitrary (in principle)

So deeper GNN ≈ larger receptive field on the graph.

---

### 2.6 Neighborhood aggregation (basic form)

**Basic recipe each layer:**
1. **average** (or sum) messages from neighbors
2. apply a neural net / nonlinearity
3. usually also mix in a **self** connection / self-transform

**Deep encoder update (lecture form):**

$$
h_v^{(0)} = x_v
$$

$$
h_v^{(l+1)}
=
\sigma\Bigg(
W^{(l)}\sum_{u\in N(v)}\frac{h_u^{(l)}}{|N(v)|}
+
B^{(l)} h_v^{(l)}
\Bigg)
,\quad
l = 0,\ldots,L-1
$$

$$
z_v = h_v^{(L)}
$$

| Piece | Role |
|-------|------|
| $\sum_{u\in N(v)} h_u^{(l)} / \|N(v)\|$ | average neighbor embeddings (order-invariant) |
| $W^{(l)}$ | weights for **neighborhood** aggregation |
| $B^{(l)}$ | weights for **self** transform |
| $\sigma$ | nonlinearity (e.g. ReLU) |
| $z_v$ | final node embedding after $L$ layers |

**Trainable parameters:** the shared matrices $\{W^{(l)}, B^{(l)}\}_l$ — **not** one vector per node.

---

### 2.7 Order / permutation invariance (critical)

Graphs have **no canonical node order**.

1. **Node permutation of the whole graph:** reordering vertices must not change the meaning of embeddings of corresponding nodes.
2. **Neighbor permutation inside aggregation:**  
   $\operatorname{Aggr}(\bullet,\bullet,\bullet)$ must be the **same** under any reordering of neighbors.

If aggregation is *not* order-invariant (e.g. a plain sequence RNN over an arbitrary neighbor list), isomorphic / identical neighborhoods can get different embeddings — bad.

**Safe aggregators:** sum, mean, max, ... (symmetric functions).

---

### 2.8 Matrix formulation (efficient implementation)

Stack all node embeddings at layer $l$:

$$
H^{(l)} = \big[h_1^{(l)}\;\cdots\;h_{|V|}^{(l)}\big]^\top
$$

Neighbor-sum via adjacency:

$$
\sum_{u\in N(v)} h_u^{(l)} = (A H^{(l)})_{v,:}
$$

Degree matrix $D$: diagonal with $D_{vv} = \deg(v) = |N(v)|$.  
Then mean aggregation:

$$
D^{-1} A H^{(l)}
\quad\text{(row-normalized neighbor average)}
$$

*(For directed graphs, use the appropriate adjacency orientation / normalization.)*

**Full layer in matrix form:**

$$
H^{(l+1)}
=
\sigma\Big(
\tilde{A}\, H^{(l)}\, W^{(l)}
+
H^{(l)}\, B^{(l)}
\Big)
,\qquad
\tilde{A} = D^{-1}A
$$

- **Red term:** neighborhood aggregation  
- **Blue term:** self transformation  
- $\tilde{A}$ is **sparse** → efficient sparse matmul on large graphs  

**Caveat:** not every fancy aggregator admits a clean sparse-matrix form.

---

### 2.9 How to train a GNN

Node embedding $z_v$ is a **function of the input graph** (structure + features).

#### Supervised

$$
\min_{\Theta}\; \mathcal{L}\big(y, f(z_v)\big)
$$

- regression → e.g. L2  
- classification → e.g. cross-entropy  

**Node classification example (binary, lecture form):**

$$
\mathcal{L}
=
\sum_{v\in V}
\Big[
y_v \log \sigma(z_v^\top \theta)
+
(1-y_v)\log\big(1-\sigma(z_v^\top \theta)\big)
\Big]
$$

Example task: safe vs toxic drug in a drug–drug interaction network.

#### Unsupervised

No node labels → use **graph structure as supervision**: "similar nodes should have similar embeddings."

$$
\mathcal{L}
=
\sum_{(u,v)}
\text{CE}\big(y_{uv},\; \operatorname{DEC}(z_u, z_v)\big)
$$

- $y_{uv}=1$ if $u,v$ are "similar"
- $\operatorname{DEC}$: decoder, often **inner product**
- similarity can come from random walks (node2vec / DeepWalk / struc2vec), matrix factorization, proximity, ...

---

### 2.10 Model-design checklist (4 steps)

1. **Define neighborhood aggregation** (how messages combine across layers)
2. **Define a loss** on the embeddings (supervised or unsupervised)
3. **Train** on batches of nodes (= batches of computation graphs)
4. **Generate embeddings** for nodes as needed — including nodes **never seen in training**

---

### 2.11 Inductive capability (why GNNs beat shallow lookups)

**Shared aggregation parameters** $W^{(l)}, B^{(l)}$ for **all** nodes:

- # parameters is **sublinear** in $|V|$ (does not grow one embedding per node)
- can **generalize to unseen nodes**
- can even **generalize to entirely new graphs** (same feature space / related domain)

**Examples from slides:**
- train on protein interactions in organism A → embed proteins in organism B
- social / content platforms (Reddit, YouTube, Scholar): new users/papers arrive continuously; embed them **on the fly** from local neighborhood + features

This is the big win vs shallow embedding tables (transductive only).

---

## 3. Lecture summary & what's next

**Recap:**
- GNNs build node embeddings by **aggregating neighborhood information** layer by layer
- **Main design axis** across GNN architectures: *how* they aggregate / transform messages
- Training = standard deep learning (forward, loss, backprop, SGD/Adam) on graph-defined computation graphs
- Shared params → **inductive** learning

**Next (announced in deck):** Graph Convolutional Networks (GCN) and **GraphSAGE** in more detail  
(see also reading: *Inductive Representation Learning for Large Graphs*)

---

## Quick formula cheat sheet

| Concept | Formula / fact |
|---------|----------------|
| Train objective | $\min_\Theta \mathcal{L}(y, f(x))$ |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ |
| Cross-entropy | $-\sum_i y_i \log \hat{y}_i$ |
| GD step | $\Theta \leftarrow \Theta - \eta\nabla_\Theta\mathcal{L}$ |
| ReLU | $\max(x,0)$ |
| MLP layer | $\sigma(Wx+b)$ |
| GNN init | $h_v^{(0)}=x_v$ |
| GNN layer (mean + self) | $h_v^{(l+1)}=\sigma\big(W^{(l)}\sum_{u\in N(v)} h_u^{(l)}/\|N(v)\| + B^{(l)} h_v^{(l)}\big)$ |
| Final embed | $z_v = h_v^{(L)}$ |
| Matrix mean-agg | $\tilde{A}=D^{-1}A$, then $H^{(l+1)}=\sigma(\tilde{A} H^{(l)} W^{(l)} + H^{(l)} B^{(l)})$ |
| Hop radius | layer $k$ ≈ $k$-hop neighborhood |

---

## Mental model (one paragraph)

A GNN is "CNN thinking on irregular graphs": each node repeatedly pulls information from its neighbors, mixes it with its own state through shared neural weights, and after $L$ rounds holds a vector that summarizes its $L$-hop context. Because the same $W,B$ are reused everywhere, the model stays small, order-invariant (with the right aggregator), and can embed new nodes or new graphs — unlike a giant lookup table over nodes.
