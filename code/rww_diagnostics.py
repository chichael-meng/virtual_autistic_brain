"""
rww_diagnostics.py  --  checks for the virtual_autistic_brain RWW pipeline
Run from the repo's code/ folder (same relative paths as rww_fit_model.ipynb).

    python rww_diagnostics.py results              # re-score saved runs without the diagonal
    python rww_diagnostics.py sc   DLPFC A4602 1   # SC density / ordering / SC-FC baseline for one scan
    python rww_diagnostics.py atlas DLPFC A4602 1  # voxels per parcel after resampling to the BOLD grid
    python rww_diagnostics.py retest               # ses-01 vs ses-02 FC similarity (noise ceiling)
    (SC similarity between sessions: python sc_prep.py retest ../data Schaefer100+Tian+Suit)

In model_fit(), replace the SC lines (loadtxt ... log1p/norm) with
    sc = prepare_sc(participant_sc_dir(ses_path, parcellation), parcellation, n_nodes=max_parcels)
-- see sc_prep.py (SIFT2, cortex only, all edges, no log1p, unit norm, one SC per participant).

In the notebook, also use:
    r_offdiag(fc_emp, fc_test)          # the metric to report
    sim_checks(F.output_sim.bold_test)  # is the simulated BOLD just noise?
    loss_breakdown(model, F)            # which loss term drives the gradients
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

N_CTX = 100  # Schaefer-100, labels 1-50 LH, 51-100 RH (7-network order)
EXPECTED_NODES = {"Schaefer100+Tian+Suit": 184, "Schaefer200+Tian+Suit": 284, "Schaefer400+Tian+Suit": 484}


# ---------------------------------------------------------------- metric
def r_offdiag(A, B):
    """Pearson r between two FC matrices using the upper triangle only (no diagonal).
    Pairs with NaN (e.g. from an empty parcel) are dropped, with a warning."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    iu = np.triu_indices(A.shape[0], 1)
    a, b = A[iu], B[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.all():
        print(f"  [r_offdiag] dropped {np.sum(~ok)} NaN edge(s) -- check for empty parcels")
    return np.corrcoef(a[ok], b[ok])[0, 1]


def noise_sd(T):
    """SD of off-diagonal correlations between T-point independent white-noise series."""
    return 1.0 / np.sqrt(T - 1)


# ---------------------------------------------------------------- 1. saved results
def check_results(res_dir="../res"):
    rows = []
    for f in sorted(glob.glob(os.path.join(res_dir, "*", "*", "*_fc_emp.csv"))):
        ft = f.replace("_fc_emp.csv", "_fc_test.csv")
        if not os.path.exists(ft):
            continue
        E, S = np.loadtxt(f), np.loadtxt(ft)
        iu = np.triu_indices(E.shape[0], 1)
        rows.append(dict(run=os.path.basename(f)[:-11], n=E.shape[0],
                         r_with_diag=np.corrcoef(E.ravel(), S.ravel())[0, 1],
                         r_offdiag=r_offdiag(E, S),
                         sim_fc_sd=S[iu].std(), sim_fc_mean=S[iu].mean()))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nmean by parcellation size:")
    print(df.groupby("n")[["r_with_diag", "r_offdiag", "sim_fc_sd"]].mean().round(3))
    print("If sim_fc_sd ~= 1/sqrt(T-1) (0.076 for T=176), the simulated FC has no structure.")
    return df


# ---------------------------------------------------------------- 2. structural connectome
def file_paths(stim_site, sub_id, parcellation, session, gsr=True, weight="sift2"):
    """Same logic as the notebook's file_paths(); the SC file is found by sc_prep (SIFT2 by default)."""
    from sc_prep import find_sc_file
    stim_type = "Sham" if ("B" in sub_id or "S" in sub_id) else "Active"
    ses_path = f"../data/{stim_site}/{stim_type}/{sub_id}/ses-0{session}"
    atlas_path = f"{ses_path}/atlases/{sub_id}_{parcellation}.nii.gz"
    if stim_site in ("DLPFC", "RIFG"):
        bold_path = f"{ses_path}/REST/sub-{sub_id[1:]}_ses-0{session}_{'' if gsr else 'no'}GSR_Filtered_4DVolume.nii"
    else:
        bold_path = f"{ses_path}/REST/{sub_id[1:]}_0{session}_R_MEICA_T1c_medn_nlw.nii.gz"
    try:
        sc_path = find_sc_file(f"{ses_path}/dMRI", parcellation, weight)
    except FileNotFoundError:
        sc_path = None
    return sc_path, bold_path, atlas_path


def load_sc(sc_path, parcellation=None):
    """Load a cm_nodes_*.csv (count or SIFT2). These files carry one extra first row/column: label 0,
    the streamlines that ended outside every parcel (MRtrix -keep_unassigned). Drop it so that
    SC index k-1 matches atlas label k, as the BOLD time series do."""
    M = np.loadtxt(sc_path, delimiter=",")
    exp = EXPECTED_NODES.get(parcellation)
    if exp is not None and M.shape[0] == exp + 1:
        print(f"  [load_sc] {M.shape[0]}x{M.shape[1]} for {exp} labels: dropping row/col 0 (unassigned streamlines)")
        M = M[1:, 1:]
    elif exp is not None and M.shape[0] != exp:
        print(f"  [load_sc] WARNING: {M.shape} does not match {exp} labels -- check node order by hand")
    return M


def prepare_sc(sc_path, parcellation, max_parcels, keep=None, log=False):
    """Kept for compatibility. sc_prep.prepare_sc is the main version (it also picks SIFT2 vs count
    and cortex vs whole atlas); this wrapper gives labels 1..max_parcels from one file."""
    from sc_prep import prepare_sc as _prepare
    return _prepare(None, parcellation, n_nodes=max_parcels, region="all", keep=keep, log=log, path=sc_path)


def sc_summary(sc_full, fc_emp=None, n=N_CTX):
    sc = np.array(sc_full, float)[:n, :n]
    asym = np.abs(sc - sc.T).max() / (np.abs(sc).max() + 1e-12)
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0)
    iu = np.triu_indices(n, 1)
    w = sc[iu]
    out = dict(density=(w > 0).mean(), max_asymmetry=asym,
               nonzero_min=w[w > 0].min(), nonzero_median=np.median(w[w > 0]), max=w.max())
    # share of the total weight in the strongest 20% of edges, on the weights the model gets
    # (raw, unit-normed). SIFT2 in our data: ~0.91-0.95, so the weak edges add little coupling.
    sw = np.sort(w)[::-1]
    out["top20pct_share_of_weight"] = sw[: int(0.2 * len(sw))].sum() / sw.sum()
    # hemisphere structure: Schaefer order puts LH = 0..49, RH = 50..99
    h = n // 2
    within = np.r_[sc[:h, :h][np.triu_indices(h, 1)], sc[h:, h:][np.triu_indices(h, 1)]].mean()
    between = sc[:h, h:].mean()
    out["within_over_between_hemisphere"] = within / (between + 1e-12)
    if fc_emp is not None:
        out["rank r(SC, empFC)"] = r_rank(sc, fc_emp[:n, :n])
        out["r(SC, empFC)"] = r_offdiag(sc, fc_emp[:n, :n])
    return out


def r_rank(A, B):
    """Spearman r over the upper triangle: does not depend on the units of the SC (count, SIFT2, x mu)."""
    from sc_prep import spearman
    iu = np.triu_indices(A.shape[0], 1)
    a, b = np.asarray(A, float)[iu], np.asarray(B, float)[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    return spearman(a[ok], b[ok])


def check_sc(stim_site, sub_id, session, parcellation="Schaefer100+Tian+Suit", gsr=True, weight="sift2"):
    sc_path, bold_path, atlas_path = file_paths(stim_site, sub_id, parcellation, session, gsr, weight)
    if sc_path is None:
        print(f"no {weight} connectome for {sub_id} ses-0{session}")
        return
    raw = np.loadtxt(sc_path, delimiter=",")
    print(f"{sc_path}: shape {raw.shape}")
    fc_emp = extract_fc(bold_path, atlas_path)
    # node-order check: which slice lines up with the atlas labels used for the BOLD?
    for off in (0, 1):
        s = raw[off:off + N_CTX, off:off + N_CTX].astype(float)
        s = 0.5 * (s + s.T); np.fill_diagonal(s, 0)
        print(f"  rows {off}-{off + N_CTX - 1}: rank r(SC, empirical FC) = {r_rank(s, fc_emp):.3f}")
    sc = load_sc(sc_path, parcellation)
    for k, v in sc_summary(sc, fc_emp).items():
        print(f"  {k:32s} {v:.3f}")
    print("Read-out: rows 1-100 should beat rows 0-99 (row 0 = unassigned streamlines). Within/between"
          " hemisphere ratio should be well above 1 if rows 1-100 are Schaefer LH then RH. A rank r(SC, empFC)"
          " of ~0.2-0.35 is the no-model baseline; near 0 suggests SC rows are not in atlas order."
          " Median edge < 1 means the file is multiplied by mu, >= 1 streamline units: both fine after unit norm.")


# ---------------------------------------------------------------- 3. atlas on the BOLD grid
def resampled_atlas(atlas_path, bold_path, n=N_CTX):
    from nilearn import image
    bold = image.load_img(bold_path)
    atlas = image.resample_to_img(image.load_img(atlas_path), image.index_img(bold, 0), interpolation="nearest")
    a = np.asarray(image.get_data(atlas)).astype(int)
    a[a > n] = 0
    return bold, atlas, a


def extract_fc(bold_path, atlas_path, n=N_CTX):
    from nilearn import image
    bold, atlas, a = resampled_atlas(atlas_path, bold_path, n)
    data = np.asarray(image.get_data(bold), dtype=float)
    ts = np.stack([data[a == k].mean(0) if (a == k).any() else np.full(data.shape[-1], np.nan)
                   for k in range(1, n + 1)], axis=1)      # time x parcels
    return np.corrcoef(ts.T)


def check_atlas(stim_site, sub_id, session, parcellation="Schaefer100+Tian+Suit", gsr=True, png=None):
    from nilearn import image, plotting
    _, bold_path, atlas_path = file_paths(stim_site, sub_id, parcellation, session, gsr)
    bold, atlas, a = resampled_atlas(atlas_path, bold_path)
    counts = np.array([(a == k).sum() for k in range(1, N_CTX + 1)])
    mean_img = image.mean_img(bold)
    m = np.asarray(image.get_data(mean_img))
    in_brain = np.array([(m[a == k] > np.percentile(m, 50)).mean() if counts[k - 1] else 0 for k in range(1, N_CTX + 1)])
    print(f"voxels per parcel on the BOLD grid: min {counts.min()}, median {int(np.median(counts))}")
    print(f"  empty parcels: {(np.where(counts == 0)[0] + 1).tolist()}")
    print(f"  parcels with <10 voxels: {(np.where(counts < 10)[0] + 1).tolist()}")
    print(f"  parcels with <50% voxels inside bright brain signal: {(np.where(in_brain < 0.5)[0] + 1).tolist()}")
    png = png or f"atlas_check_{sub_id}_ses-0{session}.png"
    plotting.plot_roi(atlas, bg_img=mean_img, display_mode="ortho", output_file=png)
    print(f"  overlay saved to {png} -- look at it.")


# ---------------------------------------------------------------- 4. noise ceiling
def check_retest(res_dir="../res", n=N_CTX):
    pat = re.compile(r"(\w+?)_ses-0(\d)_.*?_parcels(\d+)_epoch\d+(_GSR|_noGSR)?_fc_emp\.csv$")
    fcs = {}
    for f in glob.glob(os.path.join(res_dir, "*", "*", "*_fc_emp.csv")):
        m = pat.search(os.path.basename(f))
        if m and int(m.group(3)) == n:
            fcs[(m.group(1), int(m.group(2)), m.group(4) or "")] = np.loadtxt(f)
    for (s, ses, prep), E in sorted(fcs.items()):
        if ses == 1 and (s, 2, prep) in fcs:
            print(f"  {s:9s}{prep:7s} ses-01 vs ses-02 r = {r_offdiag(E, fcs[(s, 2, prep)]):.3f}")
    print("This is roughly the ceiling for any model fitted to one 7-min session.")


# ---------------------------------------------------------------- 5. simulated output
def sim_checks(bold_sim):
    """bold_sim: nodes x time (F.output_sim.bold_test)."""
    B = np.asarray(bold_sim)
    T = B.shape[1]
    fc = np.corrcoef(B)
    iu = np.triu_indices(B.shape[0], 1)
    ac1 = np.nanmean([np.corrcoef(b[:-1], b[1:])[0, 1] for b in B])
    print(f"sim BOLD lag-1 autocorrelation {ac1:+.2f} (compare with your empirical BOLD; near 0 means white noise)")
    print(f"sim FC off-diagonal SD {fc[iu].std():.3f} vs pure-noise SD {noise_sd(T):.3f}; mean {fc[iu].mean():+.3f}")


def loss_breakdown(model, F, n_batches_burn=10):
    """Size of the FC term vs the state-prior (loss_EI) term, and their gradients, at the model's operating point."""
    import torch
    N, H, B = model.node_size, model.hidden_size, model.batch_size
    X = torch.tensor(0.45 * np.random.uniform(0, 1, (N, 6)) + np.array([0, 0, 0, 1.0, 1.0, 1.0]), dtype=torch.float32)
    hE = torch.tensor(np.random.uniform(0.5, 2, (N, 500)), dtype=torch.float32)
    ext = torch.zeros(N, H, B)
    with torch.no_grad():
        for _ in range(n_batches_burn):
            nb, hE = model(ext, torch.randn(N, H, B, model.input_size), torch.randn(N, B), X, hE)
            X = nb["current_state"]
    nb, _ = model(ext, torch.randn(N, H, B, model.input_size), torch.randn(N, B), X, hE)
    emp = torch.tensor(F.ts[0][:, :B], dtype=torch.float32)
    fc_term = 1.0 * F.cost.cost_eff(nb["bold_batch"], emp)
    ei = 0
    for s, mname, vname in [("E_batch", "E_m", "E_v"), ("I_batch", "I_m", "I_v"), ("f_batch", "f_m", "f_v"), ("v_batch", "v_m", "v_v")]:
        v, m = getattr(model, vname), getattr(model, mname)
        ei = ei + torch.mean(v * (nb[s] - m) ** 2) - torch.log(v)
    print(f"FC term {fc_term.item():.3f}   state-prior term {ei.item():.1f}   "
          f"(E mean {nb['E_batch'].mean().item():.3f} vs E_m {model.E_m.item():.2f}; I mean {nb['I_batch'].mean().item():.3f} vs I_m {model.I_m.item():.2f})")
    names = [k for k in ["g", "I_0", "g_EE", "g_IE", "g_EI"] if isinstance(getattr(model, k, None), torch.nn.Parameter)]
    for label, term in [("FC term", fc_term), ("state prior", ei)]:
        gr = torch.autograd.grad(term, [getattr(model, k) for k in names], retain_graph=True, allow_unused=True)
        print(f"  |grad| from {label:11s}: " + ", ".join(f"{k} {0 if g is None else g.abs().item():.2e}" for k, g in zip(names, gr)))


# ---------------------------------------------------------------- SC options to try
def threshold_proportional(sc, keep=0.15):
    """Keep the strongest `keep` fraction of edges (symmetric), zero the rest. Sensitivity checks only;
    for anything compared across scans use one group mask (sc_prep.consistency_mask)."""
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0)
    iu = np.triu_indices(sc.shape[0], 1)
    thr = np.quantile(sc[iu], 1 - keep)
    out = np.where(sc >= thr, sc, 0.0)
    np.fill_diagonal(out, 0)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "results"
    if cmd == "results":
        check_results()
    elif cmd == "sc":
        check_sc(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "atlas":
        check_atlas(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "retest":
        check_retest()
    else:
        print(__doc__)
