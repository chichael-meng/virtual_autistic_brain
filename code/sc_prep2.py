"""
sc_prep.py -- load and prepare structural connectomes for the RWW model
(virtual_autistic_brain). Put it in code/ next to rww_diagnostics.py.

What it fixes and what it decides
  * Every cm_nodes_*.csv from the MRtrix pipeline has ONE EXTRA first row/column: label 0, the
    streamlines that ended outside every parcel. The old code sliced [0:N, 0:N], which kept that row
    and put every SC node one label away from its BOLD time series.
  * The matrices are 99-100% dense (probabilistic tractography). log1p + Frobenius norm, the
    GriffithsLab recipe for sparse deterministic matrices, flattens them into near-uniform coupling.
  * Units differ between sites: the DLPFC (and RIFG) SIFT2 files are raw sums of streamline weights
    (median edge ~300-600); the old pSTS files were already multiplied by mu (median edge ~0.01-0.03).
    Chun-Hung's re-exported files are all raw, with each participant's mu in tractogram_mu.txt.
    prepare_sc multiplies a raw SIFT2 matrix by that mu when the file is in the dMRI folder (fibre
    bundle capacity units; Smith et al. 2022, Aperture Neuro) and never scales a file twice. For the
    model input this makes no difference (the unit norm cancels any constant), but it keeps every
    matrix in the same units for anything that uses absolute values. Unit (Frobenius) norm makes
    them comparable. log1p would not: it flattens the DLPFC matrices (model r ~0.05) but only trims the
    strongest edges of the pSTS ones (model r unchanged), so the two sites would get different
    preprocessing. So: no log1p before the norm.
  * Defaults: SIFT2 weights, cortex only, all edges, no log1p, unit norm. SIFT2 is chosen because its
    weights estimate fibre density (Smith et al. 2015), not because it fits better: counts without
    log1p fit the same. Do not divide counts by length (Yeh et al. 2016).
  * No threshold by default. Edges outside the strongest 20% carry only ~5-10% of the total SIFT2
    weight, keeping only the top 20% did not change the model fit, and per-person thresholds change
    ~7% of the kept edges between two sessions of the same person. If you need a sparse version
    (sensitivity analysis), use ONE group mask for everybody (consistency_mask, Roberts et al. 2017),
    not a separate proportional threshold per scan.
  * Hold the SC fixed within a participant: use one connectome (baseline) for every session, so a
    change in fitted parameters between sessions comes from the FC and not from re-tracking noise.

In model_fit(), replace the SC block (np.loadtxt ... np.log1p ... norm) with:

    from sc_prep import prepare_sc, participant_sc_dir
    sc = prepare_sc(participant_sc_dir(ses_path, parcellation), parcellation, n_nodes=max_parcels)
    # dist is unused while the delayed term is multiplied by 0 in forward();
    # if delays are ever switched on, use prepare_length(...), not 1/sc:
    dist = np.zeros_like(sc)

Command line (run from code/):
    python sc_prep.py check  ../data                         # empty / malformed files, units per file
    python sc_prep.py retest ../data Schaefer100+Tian+Suit   # SC similarity between sessions vs between people
    python sc_prep.py mask   ../data Schaefer100+Tian+Suit --keep 0.2 --out ../res/sc_mask_consistency20.csv
    python sc_prep.py group  ../data/DLPFC Schaefer100+Tian+Suit --out ../res/sc_group_DLPFC.csv
    python sc_prep.py compare ../data/DLPFC/Active/A4608/ses-01/dMRI Schaefer100+Tian+Suit \\
                              --fc ../res/DLPFC/A4608/<run>_fc_emp.csv [--mask <mask.csv>] [--model]

Note: the SC-only r is NOT how to pick the model input. Dense log1p SC can have a decent SC-only r and
still drive the model to unstructured noise. Compare inputs with --model.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

N_SUBCORTEX = 50   # Tian S3
N_CEREBELLUM = 34  # SUIT


# ------------------------------------------------------------------ parcellation bookkeeping
def n_cortex(parcellation):
    """'Schaefer200+Tian+Suit' -> 200."""
    m = re.search(r"Schaefer(\d+)", parcellation)
    if not m:
        raise ValueError(f"cannot read the Schaefer resolution from {parcellation!r}")
    return int(m.group(1))


def n_labels(parcellation):
    """Labels in the atlas image: cortex + Tian S3 (50) + SUIT (34), e.g. 100 -> 184."""
    return n_cortex(parcellation) + N_SUBCORTEX + N_CEREBELLUM


def node_index(parcellation, region="cortex"):
    """0-based node indices (atlas label k -> index k-1) for 'cortex', 'subcortex', 'cerebellum', 'all'."""
    c = n_cortex(parcellation)
    blocks = {"cortex": np.arange(c),
              "subcortex": np.arange(c, c + N_SUBCORTEX),
              "cerebellum": np.arange(c + N_SUBCORTEX, c + N_SUBCORTEX + N_CEREBELLUM),
              "all": np.arange(n_labels(parcellation))}
    return blocks[region]


# ------------------------------------------------------------------ finding and loading files
def find_sc_file(dmri_dir, parcellation, weight="sift2"):
    """Find cm_nodes_<parcellation>*<weight>*.csv, whatever the naming variant
    (cm_nodes_X_SIFT2.csv, cm_nodes_X.nii.gz_sift2.csv, cm_nodes_X.nii.gz_count.csv, ...)."""
    pat = os.path.join(dmri_dir, f"cm_nodes_{glob.escape(parcellation)}*.csv")
    hits = [f for f in glob.glob(pat) if weight.lower() in os.path.basename(f).lower()]
    if not hits:
        raise FileNotFoundError(f"no {weight} connectome for {parcellation} in {dmri_dir}")
    hits.sort(key=len)
    return hits[0]


def load_connectome(path, parcellation):
    """Load one connectome CSV and return an (n_labels x n_labels) matrix whose index k-1 is atlas label k.
    Drops the unassigned row/column 0 when present; refuses empty or wrongly sized files."""
    if os.path.getsize(path) == 0:
        raise ValueError(f"{path} is empty (0 bytes) -- ask for the file again")
    M = np.loadtxt(path, delimiter=",")
    n = n_labels(parcellation)
    if M.shape == (n + 1, n + 1):
        M = M[1:, 1:]                       # drop label 0 = unassigned streamlines
    elif M.shape != (n, n):
        raise ValueError(f"{path}: shape {M.shape}, expected {(n + 1, n + 1)} or {(n, n)} for {parcellation}")
    return M.astype(float)


def find_mu(dmri_dir):
    """The participant's SIFT2 proportionality coefficient (tcksift2 -out_mu), e.g. tractogram_mu.txt,
    or None if the folder has no *mu.txt file."""
    if not dmri_dir:
        return None
    hits = [f for f in glob.glob(os.path.join(dmri_dir, "*.txt")) if os.path.basename(f).lower().endswith("mu.txt")]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(f"more than one mu file in {dmri_dir}: {hits}")
    return float(np.loadtxt(hits[0]).ravel()[0])


def scale_by_mu(M, dmri_dir, path=""):
    """Multiply a raw SIFT2 connectome by the participant's mu (fibre bundle capacity units).
    Leaves the matrix alone if there is no mu file, or if it already looks scaled (median non-zero
    edge < 1, as in the old pSTS files), so nothing is multiplied twice. Returns (matrix, mu or None)."""
    mu = find_mu(dmri_dir)
    if mu is None:
        return M, None
    w = M[M > 0]
    if w.size and np.median(w) < 1:
        print(f"[sc_prep] {path or dmri_dir}: median edge {np.median(w):.3g} < 1, already multiplied by mu; not scaling again")
        return M, None
    return M * mu, mu


def _block(M, parcellation, region="cortex", n_nodes=None):
    idx = node_index(parcellation, region) if isinstance(region, str) else np.asarray(region)
    if n_nodes is not None:
        idx = idx[:n_nodes]
    S = M[np.ix_(idx, idx)]
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    return S


def participant_sc_dir(ses_path, parcellation, weight="sift2", prefer="ses-01"):
    """dMRI folder whose connectome is used for EVERY session of this participant: the baseline
    session if its file is usable, otherwise the earliest session that has one (printed).
    ses_path is the session folder, e.g. ../data/DLPFC/Active/A4608/ses-02."""
    sub_dir = os.path.dirname(os.path.normpath(ses_path))
    sessions = sorted(d for d in os.listdir(sub_dir) if re.fullmatch(r"ses-\d+", d))
    for s in [prefer] + [s for s in sessions if s != prefer]:
        d = os.path.join(sub_dir, s, "dMRI")
        try:
            if os.path.getsize(find_sc_file(d, parcellation, weight)) > 0:
                if s != prefer:
                    print(f"[sc_prep] {sub_dir}: no usable {prefer} {weight} file; using {s} for all sessions")
                return d
        except (FileNotFoundError, OSError):
            continue
    raise FileNotFoundError(f"no usable {weight} connectome for {parcellation} in any session of {sub_dir}")


# ------------------------------------------------------------------ preparing the model input
def threshold_proportional(S, keep):
    """Keep the strongest `keep` fraction of off-diagonal edges of ONE matrix. For sensitivity checks
    only: the kept edge set differs between people and between sessions (use consistency_mask instead
    when scans will be compared)."""
    S = S.copy()
    iu = np.triu_indices(S.shape[0], 1)
    thr = np.quantile(S[iu], 1 - keep)
    S[S < thr] = 0.0
    np.fill_diagonal(S, 0.0)
    return S


def prepare_sc(dmri_dir, parcellation, n_nodes=None, region="cortex", weight="sift2",
               keep=None, log=False, mask=None, path=None):
    """Structural connectome ready for RNNWWD: correct node order, chosen region, symmetric,
    zero diagonal, optional group mask or proportional threshold, optional log1p, Frobenius norm = 1.

    region : 'cortex' (recommended), 'all', or a list of 0-based indices
    n_nodes: optional cap, e.g. max_parcels=100 with region='all' keeps labels 1-100
    weight : 'sift2' (recommended) or 'count'
    mask   : None (default, all edges) or a 0/1 matrix from consistency_mask(), same size as the block
    keep   : per-scan proportional threshold (sensitivity checks only), or None
    log    : log1p before normalizing. Leave False (it depends on units; see the module notes)
    """
    path = path or find_sc_file(dmri_dir, parcellation, weight)
    M = load_connectome(path, parcellation)
    if weight.lower() == "sift2":
        M, _ = scale_by_mu(M, dmri_dir or os.path.dirname(path), path)
    S = _block(M, parcellation, region, n_nodes)
    if mask is not None:
        mask = np.asarray(mask, float)
        if mask.shape != S.shape:
            raise ValueError(f"mask shape {mask.shape} does not match SC block {S.shape}")
        S = S * (mask > 0)
    if keep is not None:
        S = threshold_proportional(S, keep)
    if log:
        S = np.log1p(S)
    return S / np.linalg.norm(S)


def prepare_length(dmri_dir, parcellation, region="cortex", n_nodes=None, path=None):
    """Mean streamline length per edge (mm), same node order -- for delays or a distance baseline."""
    path = path or find_sc_file(dmri_dir, parcellation, "length")
    return _block(load_connectome(path, parcellation), parcellation, region, n_nodes)


# ------------------------------------------------------------------ several scans at once
def collect_sc(root, parcellation, session="ses-01", region="cortex", weight="sift2"):
    """Raw symmetric matrices (zero diagonal) for every <root>/**/<session>/dMRI with a usable file.
    Returns {participant folder: matrix}. Unusable files are listed and skipped."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "**", session, "dMRI"), recursive=True)):
        try:
            M = load_connectome(find_sc_file(d, parcellation, weight), parcellation)
            if weight.lower() == "sift2":
                M, _ = scale_by_mu(M, d)
            out[os.path.dirname(os.path.dirname(d))] = _block(M, parcellation, region)
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {d}: {e}")
    return out


def consistency_mask(mats, keep=0.2):
    """Consistency-based threshold (Roberts et al. 2017): keep the edges whose weights vary least
    across people (lowest coefficient of variation), as ONE edge set used for everybody.
    mats: list of same-sized raw matrices, one per person (e.g. all baseline scans, all sites).
    Each matrix is divided by its total weight first, so unit differences (mu-scaled or not) do
    not enter the CV. Returns a symmetric 0/1 matrix; every node keeps at least one edge.
    Needs many people: masks from two random halves of 8 people overlapped only about half in our data."""
    mats = [np.asarray(m, float) for m in mats]
    n = mats[0].shape[0]
    iu = np.triu_indices(n, 1)
    if len(mats) < 20:
        print(f"[consistency_mask] only {len(mats)} matrices: the CV ranking will be noisy")
    V = np.array([m[iu] / m[iu].sum() for m in mats])
    mu, sd = V.mean(0), V.std(0, ddof=1)
    cv = np.full(mu.shape, np.inf)
    cv[mu > 0] = sd[mu > 0] / mu[mu > 0]
    sel = np.zeros(len(cv), bool)
    sel[np.argsort(cv, kind="stable")[: int(round(keep * len(cv)))]] = True
    A = np.zeros((n, n))
    A[iu] = sel
    A = A + A.T
    C = np.full((n, n), np.inf)
    C[iu] = cv
    C = np.minimum(C, C.T)
    for i in np.where(A.sum(1) == 0)[0]:             # no isolated nodes
        j = int(np.argmin(C[i]))
        A[i, j] = A[j, i] = 1.0
    return A


def group_sc(mats):
    """Group-average SC: mean of the unit-normed matrices, unit-normed again. For group-level fits
    (Track A) or as a stand-in for a participant whose own file is missing."""
    G = np.mean([m / np.linalg.norm(m) for m in mats], axis=0)
    return G / np.linalg.norm(G)


def _ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    return np.corrcoef(_ranks(a), _ranks(b))[0, 1]


def top_set(S, keep=0.2):
    iu = np.triu_indices(S.shape[0], 1)
    w = S[iu]
    return set(np.argsort(w)[::-1][: int(round(keep * len(w)))])


def dice(a, b):
    return 2 * len(a & b) / (len(a) + len(b))


def sc_retest(root, parcellation, region="cortex", weight="sift2", keep=0.2):
    """Session-to-session similarity of each participant's SC (rank correlation of all edge weights;
    overlap of the strongest `keep` fraction of edges), next to the same numbers between people."""
    by_sub = {}
    for d in sorted(glob.glob(os.path.join(root, "**", "ses-*", "dMRI"), recursive=True)):
        sub, ses = os.path.dirname(os.path.dirname(d)), os.path.basename(os.path.dirname(d))
        try:
            by_sub.setdefault(sub, {})[ses] = _block(
                load_connectome(find_sc_file(d, parcellation, weight), parcellation), parcellation, region)
        except (FileNotFoundError, ValueError):
            continue
    iu = None
    within = []
    for sub, ss in sorted(by_sub.items()):
        if len(ss) < 2:
            continue
        s = sorted(ss)
        A, B = ss[s[0]], ss[s[1]]
        iu = np.triu_indices(A.shape[0], 1)
        r, dc = spearman(A[iu], B[iu]), dice(top_set(A, keep), top_set(B, keep))
        within.append((r, dc))
        print(f"  {os.path.relpath(sub, root):32s} {s[0]} vs {s[1]}: rank r {r:.3f}, top-{int(keep * 100)}% overlap {dc:.2f}")
    firsts = [ss[sorted(ss)[0]] for ss in by_sub.values()]
    between = []
    for i in range(len(firsts)):
        for j in range(i + 1, len(firsts)):
            iu = np.triu_indices(firsts[i].shape[0], 1)
            between.append((spearman(firsts[i][iu], firsts[j][iu]), dice(top_set(firsts[i], keep), top_set(firsts[j], keep))))
    if within:
        print(f"same person, two sessions (n={len(within)}): rank r {np.mean([w[0] for w in within]):.3f}, "
              f"overlap {np.mean([w[1] for w in within]):.2f}")
    if between:
        print(f"different people (n={len(between)} pairs):    rank r {np.mean([b[0] for b in between]):.3f}, "
              f"overlap {np.mean([b[1] for b in between]):.2f}")


# ------------------------------------------------------------------ checks and comparisons
def r_offdiag(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    a, b = A[iu], B[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def check_tree(root):
    """Walk a data tree and report every connectome file that is empty or wrongly sized, plus the
    units of each usable file (median edge weight), so mixed scales are visible."""
    bad, good, scales = 0, 0, {}
    for path in sorted(glob.glob(os.path.join(root, "**", "cm_nodes_*.csv"), recursive=True)):
        name = os.path.basename(path)
        m = re.search(r"cm_nodes_(.+?)(\.nii\.gz)?_(count|sift2|length)", name, flags=re.I)
        if not m:
            continue
        parc, kind = m.group(1), m.group(3).lower()
        try:
            if os.path.getsize(path) == 0:
                raise ValueError("empty file (0 bytes)")
            raw = np.loadtxt(path, delimiter=",")
            n = n_labels(parc)
            if raw.shape not in ((n + 1, n + 1), (n, n)):
                raise ValueError(f"shape {raw.shape}, expected {n + 1} or {n}")
            good += 1
            if kind == "sift2":
                w = raw[1:, 1:] if raw.shape[0] == n + 1 else raw
                med = np.median(w[w > 0])
                has_mu = find_mu(os.path.dirname(path)) is not None
                if med < 1:
                    key = "already x mu (median edge < 1)"
                elif has_mu:
                    key = "raw, mu file present (multiplied by mu on loading)"
                else:
                    key = "raw, NO mu file (fine for the model; ask for tractogram_mu.txt)"
                scales.setdefault(key, []).append(os.path.relpath(path, root))
        except Exception as e:  # noqa: BLE001 -- report and carry on
            bad += 1
            print(f"PROBLEM  {os.path.relpath(path, root)}: {e}")
    print(f"{good} connectome files OK, {bad} with problems")
    for key, files in scales.items():
        print(f"  SIFT2 files in {key}: {len(files)} (e.g. {files[0]})")
    if len(scales) > 1:
        print("  Mixed SIFT2 scales: fine after unit norm (prepare_sc), but never apply log1p.")
        print("  Best: use only the raw re-exported files plus their tractogram_mu.txt, not the old pSTS files.")


def model_fc(sc, n_tr=1760, g=80.0, seed=0, tr=2.5):
    """FC of Michael's RNNWWD driven by `sc`, with all parameters fixed at the notebook-00 values
    (no training), from one long test run (n_tr must be a multiple of 16). Needs rww_pytorch_model.py
    on the path. Use it to compare SC inputs, not as a final fit."""
    import contextlib
    import io
    import torch
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)       # rww_pytorch_model.py sits next to this file in code/
    from rww_pytorch_model import ParamsJR, RNNWWD, Model_fitting
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = sc.shape[0]
    par = ParamsJR("WWD", g=[g, 0], g_EE=[0.5, 0], g_IE=[0.5, 0], g_EI=[0.16963709468378746, 0],
                   I_0=[0.39410256410256406, 0], std_in=[0.02, 0], mu=[2.5, 0])
    model = RNNWWD(2, n, 16, 0.05, tr, sc, np.zeros_like(sc), False, par)
    F = Model_fitting(model, np.zeros((1, n, n_tr)), 1, 1, 0.001)
    X0 = 0.45 * np.random.uniform(0, 1, (n, 6)) + np.array([0, 0, 0, 1.0, 1.0, 1.0])
    hE0 = np.random.uniform(0, 5, (n, 500))
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        F.test(X0, hE0, 20)
    return np.corrcoef(F.output_sim.bold_test)


def compare_inputs(dmri_dir, parcellation, fc_emp, region="cortex", run_model=False, mask=None):
    """SC-only (and optionally model) r for count vs SIFT2 and each preparation, plus a
    distance-only baseline. fc_emp must be in atlas label order and cover the chosen region."""
    idx = node_index(parcellation, region)
    F = np.asarray(fc_emp, float)[np.ix_(idx, idx)]
    print(f"{parcellation}, {region} ({len(idx)} nodes) -- r with empirical FC, upper triangle")
    variants = [("all edges (default)", dict()), ("own top 20%", dict(keep=0.20)), ("all edges, log1p", dict(log=True))]
    if mask is not None:
        variants.insert(1, ("group mask", dict(mask=mask)))
    for weight in ("sift2", "count"):
        try:
            path = find_sc_file(dmri_dir, parcellation, weight)
        except FileNotFoundError as e:
            print(f"  {weight:6s}: {e}")
            continue
        for label, kw in variants:
            S = prepare_sc(dmri_dir, parcellation, region=region, weight=weight, path=path, **kw)
            line = f"  {weight:6s} {label:20s} SC-only r = {r_offdiag(S, F):.3f}"
            if run_model:
                line += f"   model r (long run) = {r_offdiag(model_fc(S), F):.3f}"
            print(line, flush=True)
    try:
        L = prepare_length(dmri_dir, parcellation, region=region)
        Lm = np.where(L > 0, L, np.nan)
        print(f"  distance-only baseline: r(-length, FC) = {r_offdiag(-Lm, F):.3f}  (a model should beat this too)")
    except FileNotFoundError:
        pass


def _main():
    p = argparse.ArgumentParser(description="SC loading and checks for the RWW pipeline")
    sub = p.add_subparsers(dest="cmd")
    a = sub.add_parser("check"); a.add_argument("root")
    a = sub.add_parser("retest"); a.add_argument("root"); a.add_argument("parcellation")
    a.add_argument("--weight", default="sift2")
    a = sub.add_parser("mask"); a.add_argument("root"); a.add_argument("parcellation")
    a.add_argument("--keep", type=float, default=0.2); a.add_argument("--session", default="ses-01")
    a.add_argument("--out", required=True)
    a = sub.add_parser("group"); a.add_argument("root"); a.add_argument("parcellation")
    a.add_argument("--session", default="ses-01"); a.add_argument("--out", required=True)
    a = sub.add_parser("compare"); a.add_argument("dmri_dir"); a.add_argument("parcellation")
    a.add_argument("--fc", required=True); a.add_argument("--mask"); a.add_argument("--model", action="store_true")
    args = p.parse_args()
    if args.cmd == "check":
        check_tree(args.root)
    elif args.cmd == "retest":
        sc_retest(args.root, args.parcellation, weight=args.weight)
    elif args.cmd == "mask":
        mats = collect_sc(args.root, args.parcellation, session=args.session)
        A = consistency_mask(list(mats.values()), keep=args.keep)
        np.savetxt(args.out, A, fmt="%d", delimiter=",")
        print(f"mask from {len(mats)} scans, {int(A[np.triu_indices(len(A), 1)].sum())} edges kept -> {args.out}")
    elif args.cmd == "group":
        mats = collect_sc(args.root, args.parcellation, session=args.session)
        np.savetxt(args.out, group_sc(list(mats.values())), delimiter=",")
        print(f"group SC from {len(mats)} scans -> {args.out}")
    elif args.cmd == "compare":
        mask = np.loadtxt(args.mask, delimiter=",") if args.mask else None
        compare_inputs(args.dmri_dir, args.parcellation, np.loadtxt(args.fc), run_model=args.model, mask=mask)
    else:
        print(__doc__)


if __name__ == "__main__":
    _main()
