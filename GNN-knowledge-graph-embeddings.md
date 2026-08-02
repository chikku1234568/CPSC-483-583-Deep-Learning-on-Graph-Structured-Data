# Knowledge Graph Embeddings — Notes from `16-kg.pdf`

**Course:** CPSC 483 / 583 — Deep Learning on Graph-Structured Data  
**Instructor:** Rex Ying  
**Lecture:** Knowledge Graph Embeddings

**Key Readings:**
- TransE: Translating Embeddings for Modeling Multi-relational Data
- RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
- TuckER: Tensor Factorization for Knowledge Graph Completion

---

## 1. Knowledge Graphs (KGs)

A Knowledge Graph represents knowledge as a collection of **triples**:
$$
(h, r, t) \quad \text{head entity } h \text{ has relation } r \text{ with tail entity } t
$$

**Characteristics:**
- Nodes = entities (with types)
- Edges = relations (typed)
- Heterogeneous graph (multiple node and edge types)

**Examples of real KGs:**
- Google Knowledge Graph / Knowledge Vault
- Amazon Product Graph, Facebook Graph API, LinkedIn KG
- Wikidata, Freebase, Dbpedia, YAGO, NELL

**Key properties of public KGs:**
- Extremely large (millions to billions of facts)
- Highly incomplete (many true facts are missing)

---

## 2. KG Completion Task

Because KGs are incomplete, the main task is:

> Given $(h, r)$, predict the missing $t$ (or vice versa).

This is slightly different from standard **link prediction**:
- Link prediction: predict whether an edge $(u, v)$ exists
- KG completion: given a relation type, predict the correct tail entity

---

## 3. KG Representation Learning

**Core idea:**
- Embed entities and relations into a vector space $\mathbb{R}^d$
- Learn shallow embeddings (or use GNN encoders) such that for a true triple:
  $$
  \text{embedding of } (h, r) \approx \text{embedding of } t
  $$

Key questions:
- How to embed $(h, r)$ together?
- How to define "closeness" (scoring function)?

---

## 4. TransE (Translating Embeddings)

**Intuition:** Treat relations as **translations** in embedding space.

For a true triple $(h, r, t)$:
$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
$$

**Scoring function:**
$$
f_r(h, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|
$$

- Higher score = more likely to be true
- Goal: maximize score for true triples, minimize for false ones

### Initialization (from the paper)
- Sample uniformly from $\left(-\frac{6}{\sqrt{k}}, \frac{6}{\sqrt{k}}\right)$ (Xavier-like)
- Normalize all entity and relation vectors to unit length

### Negative Sampling
For a positive triple $(h, r, t)$, create corrupted triples by replacing **either** head **or** tail (not both):
$$
S'_{(h,r,t)} = \{(h', r, t) \mid h' \in E\} \cup \{(h, r, t') \mid t' \in E\}
$$

### Training Loss (Max-Margin)
$$
\mathcal{L} = \sum_{(h,r,t) \in G} \sum_{(h',r,t') \in S'} \big[ \gamma - f_r(h,t) + f_r(h',t') \big]_+
$$

- $\gamma$ = margin hyperparameter
- Only penalizes when the corrupted triple scores higher than the valid one (within margin)

---

## 5. Connectivity Patterns in Knowledge Graphs

Real KGs contain relations with different algebraic properties:

| Pattern               | Definition                                      | Example                  |
|-----------------------|-------------------------------------------------|--------------------------|
| Symmetric             | $r(h,t) \Rightarrow r(t,h)$                     | Roommate, Spouse         |
| Antisymmetric         | $r(h,t) \Rightarrow \neg r(t,h)$                | ParentOf, Advisor        |
| Inverse               | $r_2(h,t) \Rightarrow r_1(t,h)$                 | Advisor ↔ Advisee        |
| Composition (Transitive) | $r_1(x,y) \land r_2(y,z) \Rightarrow r_3(x,z)$ | mother’s husband = father |
| 1-to-N                | One head can have many tails under same relation| StudentsOf(professor)    |

A good KG embedding method should be able to model as many of these patterns as possible.

---

## 6. What TransE Can and Cannot Model

### TransE Strengths

- **Antisymmetric relations**: $\mathbf{h} + \mathbf{r} = \mathbf{t}$ but $\mathbf{t} + \mathbf{r} \neq \mathbf{h}$ → works well ✓
- **Inverse relations**: Set $\mathbf{r}_1 = -\mathbf{r}_2$ → works ✓
- **Composition relations**: $\mathbf{r}_3 = \mathbf{r}_1 + \mathbf{r}_2$ → works ✓

### TransE Limitations (will be discussed in later parts of lecture)

- Struggles with **symmetric relations** (hard to satisfy both $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$ and $\mathbf{t} + \mathbf{r} \approx \mathbf{h}$ unless $\mathbf{r} \approx 0$)
- Cannot properly handle **1-to-N, N-to-1, N-to-N** relations (because translation is deterministic)

---

## 7. Key Takeaways So Far

- Knowledge graphs are massive, heterogeneous, and incomplete → KG completion is a core task.
- TransE is the foundational model: simple translation-based embedding $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$.
- TransE is surprisingly effective at modeling antisymmetric, inverse, and compositional relations.
- It has clear limitations on symmetric and multi-valued (1-to-N) relations, which later models (RotatE, ComplEx, TuckER, etc.) address.

---

(The lecture continues with DistMult, ComplEx, RotatE, and TuckER to overcome TransE's limitations. The extracted pages end during the TransE analysis.)

This lecture introduces the classic line of work on **shallow KG embedding models** that dominated early KG completion research.
