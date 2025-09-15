# VLP: Efficient Vector‑Based Label Propagation for Massive Low‑Rank Graphs

VLP is a reference implementation and reproducibility package for **vector‑based label propagation** on **low‑rank graphs**, where the adjacency has the form $A = VV^\top$. Instead of constructing an $n\times n$ adjacency, VLP performs propagation directly in the embedding space using two dense matrix multiplications per iteration. This reduces memory from $O(n^2)$ to $O(nd)$ and closely follows the algorithm described in the accompanying paper.

> **Scope.** The artifact evaluates **linear (numeric) label propagation** baselines (e.g., Zhu & Ghahramani–style) implemented in scikit‑learn, scikit‑network, and PyTorch Geometric. Voting‑style LP used for community detection is out of scope here.

---

## Key idea (summary)

Given node embeddings $V\in\mathbb{R}^{n\times d}$, we work with a non‑negative similarity form (e.g., normalized cosine with a bias term). Let

* **self\_loop**: $s_i = v_i^\top v_i$,
* **degree**: $k_i = \sum_j v_i^\top v_j - s_i$,
* **inv\_deg**: $1/k_i$ (row‑wise normalization).

One VLP iteration updates label matrix $Y^{(t)}\in\mathbb{R}^{n\times c}$ as:

$$
Y^{(t+1)} = \text{inv\_deg} \odot \Big( V (V^\top Y^{(t)}) - \text{self\_loop} \odot Y^{(t)} \Big),
$$

where $\odot$ denotes element‑wise operations with appropriate broadcasting. This produces the same result as adjacency‑based LP on the corresponding dense graph without materializing $A$.

---

## What’s inside

```
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ .gitattributes
├─ pyproject.toml
├─ requirements.txt
├─ environment.yml
├─ Makefile
├─ configs/
│  ├─ default.yaml
│  ├─ datasets.yaml
│  └─ experiments/
│     ├─ amazon_table2.yaml
│     ├─ yelp_table3.yaml
│     └─ taobao_table4.yaml
├─ src/
│  └─ vlp/
│     ├─ __init__.py
│     ├─ core.py
│     ├─ algo.py
│     ├─ baselines/
│     │  ├─ sklearn_lp.py
│     │  ├─ sknetwork_lp.py
│     │  └─ pyg_lp.py
│     ├─ data/
│     │  ├─ flickr.py
│     │  ├─ amazon_products.py
│     │  ├─ yelp.py
│     │  └─ taobao.py
│     ├─ eval/
│     │  ├─ runner.py
│     │  ├─ metrics.py
│     │  └─ tables.py
│     └─ utils/
│        ├─ timer.py
│        ├─ device.py
│        ├─ memory.py
│        ├─ seed.py
│        └─ logging.py
├─ scripts/
│  ├─ download_data.py
│  ├─ run_vlp.py
│  ├─ run_baselines.py
│  ├─ benchmark.py
│  ├─ make_tables.py
│  └─ make_plots.py
├─ notebooks/
│  ├─ 01_quickstart_vlp.ipynb
│  └─ 02_benchmarks.ipynb
├─ tests/
│  ├─ test_equivalence_small.py
│  ├─ test_stochasticity.py
│  ├─ test_memory_guard.py
│  └─ test_cli.py
├─ outputs/           # (gitignored)
└─ data/              # (gitignored)
```

---

## Installation

> Python 3.10+ recommended.

Using **conda**:
```bash
conda env create -f environment.yml
conda activate vlp
```

Using **pip** (on a fresh virtualenv):
```bash
pip install -r requirements.txt
```

> Note: PyTorch Geometric wheels depend on your CUDA/PyTorch versions. If needed, consult the PyG install guide to match packages for your system.

---

## Datasets

Run the data helper to download public datasets into `data/`:

```bash
python scripts/download_data.py --datasets flickr amazon yelp taobao
```

- Flickr, Amazon Products, Yelp are fetched via `torch_geometric.datasets` and cached under `data/`.
- Taobao user behavior may require access steps; the script will print instructions and perform local TF‑IDF processing if files are available.

---

## Quickstart

```bash
python scripts/run_vlp.py --dataset flickr --fraction 0.01 --iters 100 --classes 50
```

This runs VLP on a small fraction for a smoke test and prints timing plus metadata. For full benchmarks and tables, see the next section.

---

## Reproducing tables

Each paper table corresponds to a config under `configs/experiments/`:

```bash
# Amazon (Table 2)
python scripts/benchmark.py --config configs/experiments/amazon_table2.yaml

# Yelp (Table 3)
python scripts/benchmark.py --config configs/experiments/yelp_table3.yaml

# Taobao (Table 4)
python scripts/benchmark.py --config configs/experiments/taobao_table4.yaml
```

Results (CSV + Markdown tables) are written to `outputs/`. Baseline timings include graph/kernel construction; infeasible runs are reported as `X`.

---

## License

See `LICENSE` (MIT).

## Citation

Please see `CITATION.cff`.
