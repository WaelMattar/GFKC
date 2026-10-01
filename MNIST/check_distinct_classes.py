import itertools
import numpy as np
import ot
from torchvision.datasets import MNIST

ds = MNIST(".", train=True, download=True)
X = ds.data.numpy().astype(np.float64)
y = ds.targets.numpy()

# Grid coordinates in [0,1]^2 and Euclidean ground cost
g = np.arange(28) / 27.0
P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
M = ot.dist(P, P, metric="euclidean")

def to_measure(img):
    a = img.reshape(-1)
    return a / a.sum()

def w1(a, b):
    # Restrict to supports for speed; exact EMD
    ia, ib = a > 0, b > 0
    return ot.emd2(a[ia], b[ib], M[np.ix_(ia, ib)], numItermax=1_000_000)

rng = np.random.default_rng(0)
N_PAIRS = 1000  # random image pairs per class pair
idx = {k: np.flatnonzero(y == k) for k in range(10)}
mean_meas = {k: to_measure(X[idx[k]].mean(0)) for k in range(10)}

def sample_w1(i, j):
    ai, aj = rng.choice(idx[i], N_PAIRS), rng.choice(idx[j], N_PAIRS)
    return np.mean([w1(to_measure(X[p]), to_measure(X[q])) for p, q in zip(ai, aj)])

D_mean = np.zeros((10, 10))
D_samp = np.zeros((10, 10))
for i, j in itertools.combinations(range(10), 2):
    D_mean[i, j] = D_mean[j, i] = w1(mean_meas[i], mean_meas[j])
    D_samp[i, j] = D_samp[j, i] = sample_w1(i, j)
for k in range(10):
    D_samp[k, k] = sample_w1(k, k)  # within-class baseline

def report(D, name):
    print(f"\n== {name} ==")
    pairs = sorted(((D[i, j], i, j) for i, j in itertools.combinations(range(10), 2)),
                   reverse=True)
    print("Top pairs:")
    for d, i, j in pairs[:10]:
        print(f"  {i}-{j}: {d:.4f}")

    # Best triples: maximize the smallest of the three pairwise distances;
    # the shared class b is the vertex whose two edges are largest.
    triples = []
    for t in itertools.combinations(range(10), 3):
        s = min(D[p, q] for p, q in itertools.combinations(t, 2))
        b = max(t, key=lambda v: min(D[v, u] for u in t if u != v))
        a, c = [u for u in t if u != b]
        triples.append((s, a, b, c))
    triples.sort(reverse=True)
    print("Top triples:")
    for s, a, b, c in triples[:5]:
        print(f"  Set_1={{{a},{b}}}, Set_2={{{b},{c}}}   min W1 = {s:.4f}")

report(D_mean, "W1 between class-mean images")
report(D_samp, f"Mean W1 over {N_PAIRS} random image pairs")
print("\nWithin-class spread (diagonal):",
      np.round(np.diag(D_samp), 4))
