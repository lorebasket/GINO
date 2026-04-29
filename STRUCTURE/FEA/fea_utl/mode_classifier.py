"""
Curved Beam Vibration Mode Classifier
======================================
Classifies eigenmodes from a curved beam FEM model by computing
the participation ratio of each DOF group (axial, bending XY,
bending XZ, torsion) per eigenvector.

DOF convention per node (6 DOFs, Timoshenko/3D beam):
  Index 0: ux  – axial displacement (along beam tangent)
  Index 1: uy  – transverse displacement in XY plane
  Index 2: uz  – transverse displacement in XZ plane
  Index 3: θx  – torsional rotation (about beam axis)
  Index 4: θy  – bending rotation in XZ plane (curvature in XZ)
  Index 5: θz  – bending rotation in XY plane (curvature in XY)

Usage:
  1. Instantiate ModeClassifier with your eigenvectors and frequencies.
  2. Call .classify() to get a DataFrame with mode labels.
  3. Call .summary() to print a formatted table.
  4. Call .plot_participation() to visualise participation bars.
  5. Call .plot_mode_shape() to animate/plot a specific mode shape.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings

# ---------------------------------------------------------------------------
# Optional: pandas for pretty table output
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# DOF GROUP DEFINITIONS
# ---------------------------------------------------------------------------
# For a 6-DOF beam node the physical interpretation is:
#   Axial      : ux        (index 0 within the node's 6 DOFs)
#   Bending XY : uy + θz   (indices 1, 5)
#   Bending XZ : uz + θy   (indices 2, 4)
#   Torsion    : θx        (index 3)

DOF_GROUPS = {
    "Axial":       [0],
    "Bending XY":  [1, 5],
    "Bending XZ":  [2, 4],
    "Torsion":     [3],
}

GROUP_COLORS = {
    "Axial":      "#4C72B0",
    "Bending XY": "#DD8452",
    "Bending XZ": "#55A868",
    "Torsion":    "#C44E52",
    "Mixed":      "#9370DB",
}

NDOF_PER_NODE = 6   # standard 3D beam element


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _participation_ratios(phi: np.ndarray, n_nodes: int) -> dict:
    """
    Given a single eigenvector phi (length = n_nodes * 6),
    returns a dict {group_name: ratio} where ratio ∈ [0, 1]
    and all ratios sum to 1.

    The metric used is the squared L2 norm of each DOF-group's
    components, normalised by the total squared L2 norm.
    """
    phi = np.asarray(phi, dtype=float)
    phi = phi / (np.linalg.norm(phi) + 1e-300)   # unit normalise
    energy_total = np.dot(phi, phi)               # = 1.0 after normalisation

    ratios = {}
    for group, local_indices in DOF_GROUPS.items():
        # Gather global indices for this group across ALL nodes
        global_idx = []
        for node in range(n_nodes):
            for li in local_indices:
                global_idx.append(node * NDOF_PER_NODE + li)
        global_idx = np.array(global_idx, dtype=int)
        group_energy = np.sum(phi[global_idx] ** 2)
        ratios[group] = group_energy / (energy_total + 1e-300)

    return ratios


def _classify_mode(ratios: dict, threshold_dominant: float = 0.50,
                   threshold_secondary: float = 0.10) -> str:
    """
    Assign a label to a mode based on participation ratios.

    Rules (checked in order):
      1. If one group > threshold_dominant  → that group is dominant → pure mode
      2. If top-2 groups each > threshold_secondary  → "X / Y coupled" label
      3. Otherwise → "Mixed"
    
    Note: Thresholds have been adjusted to better detect weak torsional coupling.
    """
    sorted_groups = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    top_name, top_val   = sorted_groups[0]
    sec_name, sec_val   = sorted_groups[1]

    if top_val >= threshold_dominant:
        return top_name                                       # pure mode

    if top_val >= threshold_secondary and sec_val >= threshold_secondary:
        return f"{top_name} / {sec_name}"                    # coupled mode

    return "Mixed"


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class ModeClassifier:
    """
    Classifies the nature (bending XY/XZ, torsion, axial) of each
    natural frequency / eigenmode of a curved beam FEM model.

    Parameters
    ----------
    eigenvectors : ndarray, shape (n_dofs, n_modes)
        Columns are the mass-normalised eigenvectors.
    eigenvalues  : ndarray, shape (n_modes,)
        Either angular frequencies ω [rad/s]  (if freq_type='omega')
        or natural frequencies f [Hz]          (if freq_type='hz')
        or eigenvalues λ = ω²                  (if freq_type='lambda').
    n_nodes      : int
        Number of structural nodes in the model.
    freq_type    : str
        One of 'lambda' | 'omega' | 'hz'.
    threshold_dominant  : float
        Participation ratio above which a group is considered "dominant".
    threshold_secondary : float
        Minimum participation for a group to appear in a "coupled" label.
    """

    def __init__(
        self,
        eigenvectors: np.ndarray,
        eigenvalues:  np.ndarray,
        n_nodes:      int,
        freq_type:    str   = "lambda",
        threshold_dominant:  float = 0.50,
        threshold_secondary: float = 0.10,
    ):
        self.Phi   = np.asarray(eigenvectors, dtype=float)
        self.lam   = np.asarray(eigenvalues,  dtype=float)
        self.n_nodes = n_nodes
        self.freq_type = freq_type.lower()
        self.thr_dom = threshold_dominant
        self.thr_sec = threshold_secondary

        n_dofs_expected = n_nodes * NDOF_PER_NODE
        if self.Phi.shape[0] != n_dofs_expected:
            raise ValueError(
                f"eigenvectors has {self.Phi.shape[0]} rows but "
                f"n_nodes={n_nodes} implies {n_dofs_expected} DOFs."
            )

        self._compute_frequencies()
        self._results = None

    # ------------------------------------------------------------------
    def _compute_frequencies(self):
        """Convert eigenvalues to Hz and rad/s."""
        if self.freq_type == "lambda":
            self.omega = np.sqrt(np.abs(self.lam))
        elif self.freq_type == "omega":
            self.omega = np.abs(self.lam)
        elif self.freq_type == "hz":
            self.omega = 2 * np.pi * np.abs(self.lam)
        else:
            raise ValueError(f"Unknown freq_type='{self.freq_type}'")
        self.freq_hz = self.omega / (2 * np.pi)

    # ------------------------------------------------------------------
    def classify(self) -> list:
        """
        Run classification on all modes.

        Returns
        -------
        results : list of dict
            One dict per mode with keys:
            'mode', 'freq_hz', 'omega', 'label',
            'Axial', 'Bending XY', 'Bending XZ', 'Torsion'
        """
        n_modes = self.Phi.shape[1]
        results = []
        for i in range(n_modes):
            phi = self.Phi[:, i]
            ratios = _participation_ratios(phi, self.n_nodes)
            label  = _classify_mode(ratios, self.thr_dom, self.thr_sec)
            record = {
                "Mode":       i + 1,
                "f [Hz]":     self.freq_hz[i],
                "ω [rad/s]":  self.omega[i],
                "Nature":     label,
            }
            record.update({k: f"{v*100:.1f}%" for k, v in ratios.items()})
            record["_ratios"] = ratios   # keep raw for plotting
            results.append(record)
        self._results = results
        return results

    # ------------------------------------------------------------------
    def summary(self, n_decimals: int = 4):
        """Print a formatted summary table."""
        if self._results is None:
            self.classify()

        print("\n" + "=" * 78)
        print("  CURVED BEAM – VIBRATION MODE CLASSIFICATION")
        print("=" * 78)

        cols = ["Mode", "f [Hz]", "ω [rad/s]", "Nature",
                "Axial", "Bending XY", "Bending XZ", "Torsion"]

        if HAS_PANDAS:
            rows = [{c: r[c] for c in cols} for r in self._results]
            df = pd.DataFrame(rows)
            df["f [Hz]"]    = df["f [Hz]"].map(lambda x: f"{x:.{n_decimals}f}")
            df["ω [rad/s]"] = df["ω [rad/s]"].map(lambda x: f"{x:.{n_decimals}f}")
            print(df.to_string(index=False))
        else:
            # Fallback plain-text table
            fmt_header = "{:>5}  {:>12}  {:>12}  {:>20}  {:>7}  {:>10}  {:>10}  {:>8}"
            fmt_row    = "{:>5}  {:>12}  {:>12}  {:>20}  {:>7}  {:>10}  {:>10}  {:>8}"
            print(fmt_header.format(*cols))
            print("-" * 78)
            for r in self._results:
                print(fmt_row.format(
                    r["Mode"],
                    f"{r['f [Hz]']:.{n_decimals}f}",
                    f"{r['ω [rad/s]']:.{n_decimals}f}",
                    r["Nature"],
                    r["Axial"], r["Bending XY"], r["Bending XZ"], r["Torsion"],
                ))

        print("=" * 78)
        print(f"  Thresholds: dominant≥{self.thr_dom*100:.0f}%  "
              f"secondary≥{self.thr_sec*100:.0f}%")
        print("=" * 78 + "\n")

    # ------------------------------------------------------------------
    def plot_participation(self, max_modes: int = 20, figsize: tuple = (14, 6)):
        """
        Stacked-bar chart of DOF-group participation ratios for each mode.
        """
        if self._results is None:
            self.classify()

        results = self._results[:max_modes]
        n = len(results)
        groups = list(DOF_GROUPS.keys())

        matrix = np.array([[r["_ratios"][g] for g in groups] for r in results])
        labels = [f"#{r['Mode']}\n{r['f [Hz]']:.2f} Hz\n{r['Nature']}"
                  for r in results]

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(n)
        x = np.arange(n)

        for gi, g in enumerate(groups):
            bars = ax.bar(x, matrix[:, gi], bottom=bottom,
                          color=GROUP_COLORS[g], label=g, edgecolor="white", linewidth=0.5)
            # annotate bars with percentage if large enough
            for bi, (b, val) in enumerate(zip(bottom, matrix[:, gi])):
                if val > 0.08:
                    ax.text(bi, b + val / 2, f"{val*100:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
            bottom += matrix[:, gi]

        # colour-code x-tick labels by nature
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        for tick, r in zip(ax.get_xticklabels(), results):
            nature = r["Nature"]
            color = GROUP_COLORS.get(nature, GROUP_COLORS["Mixed"])
            tick.set_color(color)

        ax.set_ylabel("Participation Ratio")
        ax.set_title("Vibration Mode Participation by DOF Group", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.axhline(self.thr_dom, color="gray", linestyle="--", linewidth=0.8,
                   label=f"dominant threshold ({self.thr_dom*100:.0f}%)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def plot_mode_shape(self, mode_index: int = 0,
                        node_coords: np.ndarray = None,
                        scale: float = 1.0,
                        figsize: tuple = (14, 8)):
        """
        Plot the deformed shape of a specific mode.

        Parameters
        ----------
        mode_index   : int
            0-based index of the mode to plot.
        node_coords  : ndarray, shape (n_nodes, 3)
            Undeformed XYZ coordinates. If None, nodes are placed on
            a unit arc for illustration purposes.
        scale        : float
            Amplification factor for the deformed shape.
        """
        if self._results is None:
            self.classify()

        if node_coords is None:
            warnings.warn("No node_coords supplied – using default unit-arc geometry.")
            theta = np.linspace(0, np.pi, self.n_nodes)
            node_coords = np.column_stack([np.cos(theta), np.sin(theta),
                                           np.zeros(self.n_nodes)])

        phi = self.Phi[:, mode_index]
        # Extract translational DOFs only for visualisation
        ux = phi[0::NDOF_PER_NODE]
        uy = phi[1::NDOF_PER_NODE]
        uz = phi[2::NDOF_PER_NODE]

        # Auto-scale
        max_disp = max(np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz))) + 1e-15
        max_geom = np.max(np.abs(node_coords)) + 1e-15
        auto_scale = scale * max_geom / max_disp * 0.15

        deformed = node_coords.copy()
        deformed[:, 0] += ux * auto_scale
        deformed[:, 1] += uy * auto_scale
        deformed[:, 2] += uz * auto_scale

        r = self._results[mode_index]
        title = (f"Mode {r['Mode']}  |  {r['f [Hz]']:.4f} Hz  |  "
                 f"Nature: {r['Nature']}")

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        gs  = gridspec.GridSpec(1, 3, figure=fig)

        view_planes = [
            ("XY plane",  0, 1, "X", "Y"),
            ("XZ plane",  0, 2, "X", "Z"),
            ("YZ plane",  1, 2, "Y", "Z"),
        ]

        for idx, (plane_name, a, b, xl, yl) in enumerate(view_planes):
            ax = fig.add_subplot(gs[0, idx])
            ax.plot(node_coords[:, a], node_coords[:, b],
                    "o--", color="gray", linewidth=1.0, markersize=3,
                    label="Undeformed", zorder=1)
            ax.plot(deformed[:, a], deformed[:, b],
                    "o-", color="#E84040", linewidth=2.0, markersize=4,
                    label="Deformed", zorder=2)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(plane_name, fontsize=10)
            ax.set_aspect("equal", "datalim")
            ax.legend(fontsize=8)
            ax.grid(linestyle=":", alpha=0.5)

        fig.tight_layout()
        plt.show()


# ===========================================================================
# DEMO – synthetic curved beam example
# ===========================================================================

def _build_demo_problem(n_nodes: int = 20, n_modes: int = 10,
                         seed: int = 42) -> tuple:
    """
    Build a synthetic demo problem that mimics realistic mode shapes for a
    curved (half-circle) beam so the classifier has meaningful input.

    We construct eigenvectors analytically:
      - Modes 1,2  : bending XY  (dominant uy + θz)
      - Modes 3,4  : bending XZ  (dominant uz + θy)
      - Mode  5    : torsion     (dominant θx)
      - Mode  6    : axial       (dominant ux)
      - Mode  7,8  : bending XY  (higher harmonics)
      - Mode  9    : Bending XZ / Torsion coupling
      - Mode  10   : Mixed
    """
    rng = np.random.default_rng(seed)
    n_dofs = n_nodes * NDOF_PER_NODE
    Phi    = np.zeros((n_dofs, n_modes))
    theta  = np.linspace(0, np.pi, n_nodes)

    def sine(k):     return np.sin(k * theta)
    def cosine(k):   return np.cos(k * theta)

    # helper: fill a single DOF-lane for all nodes
    def fill(mode_col, dof_local, shape, noise=0.05):
        idx = np.arange(n_nodes) * NDOF_PER_NODE + dof_local
        Phi[idx, mode_col] = shape + noise * rng.standard_normal(n_nodes)

    # Mode 0 – bending XY (1st harmonic)
    fill(0, 1, sine(1), 0.02);   fill(0, 5, cosine(1), 0.02)

    # Mode 1 – bending XY (2nd harmonic)
    fill(1, 1, sine(2), 0.02);   fill(1, 5, cosine(2), 0.02)

    # Mode 2 – bending XZ (1st harmonic)
    fill(2, 2, sine(1), 0.02);   fill(2, 4, cosine(1), 0.02)

    # Mode 3 – bending XZ (2nd harmonic)
    fill(3, 2, sine(2), 0.02);   fill(3, 4, cosine(2), 0.02)

    # Mode 4 – torsion
    fill(4, 3, sine(1), 0.03)

    # Mode 5 – axial
    fill(5, 0, cosine(1), 0.03)

    # Mode 6 – bending XY higher harmonic
    fill(6, 1, sine(3), 0.02);   fill(6, 5, cosine(3), 0.02)

    # Mode 7 – bending XZ higher harmonic
    fill(7, 2, sine(3), 0.02);   fill(7, 4, cosine(3), 0.02)

    # Mode 8 – bending XZ / torsion coupled
    fill(8, 2, sine(2)*0.6, 0.02); fill(8, 4, cosine(2)*0.6, 0.02)
    fill(8, 3, sine(1)*0.4, 0.04)

    # Mode 9 – mixed (all groups contribute)
    fill(9, 0, sine(1)*0.25, 0.02)
    fill(9, 1, sine(2)*0.25, 0.02)
    fill(9, 2, sine(1)*0.25, 0.02)
    fill(9, 3, cosine(1)*0.25, 0.02)

    # Assign plausible eigenfrequencies (Hz)
    freqs_hz = np.array([12.3, 25.1, 31.7, 58.4, 68.2,
                         72.5, 89.0, 104.3, 115.6, 142.8])
    lambdas = (2 * np.pi * freqs_hz) ** 2    # λ = ω²

    return Phi, lambdas


def run_demo():
    print("\n" + "━"*60)
    print("  DEMO: Curved Beam Mode Classification")
    print("━"*60)

    # ----- build demo data -----
    n_nodes = 20
    Phi, lambdas = _build_demo_problem(n_nodes=n_nodes, n_modes=10)

    # ----- geometry: half-circle arc -----
    theta = np.linspace(0, np.pi, n_nodes)
    node_coords = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_nodes)])

    # ----- classify -----
    clf = ModeClassifier(
        eigenvectors = Phi,
        eigenvalues  = lambdas,
        n_nodes      = n_nodes,
        freq_type    = "lambda",
        threshold_dominant  = 0.50,
        threshold_secondary = 0.10,
    )

    clf.summary()
    clf.plot_participation(max_modes=10)

    # ----- show first 4 mode shapes -----
    for i in range(4):
        clf.plot_mode_shape(mode_index=i, node_coords=node_coords, scale=1.0)


# ===========================================================================
# HOW TO USE WITH YOUR OWN FEM DATA
# ===========================================================================
def example_usage_with_own_data():
    """
    Replace the arrays below with your own FEM eigensolution output.

    Most FEM frameworks export:
        eigenvalues  – array of λ = ω²  (or ω or f)
        eigenvectors – matrix (n_dofs × n_modes)

    Example with scipy:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(K, M)
        clf = ModeClassifier(eigenvectors, eigenvalues, n_nodes=..., freq_type='lambda')

    Example with FEniCS / OpenSees / ABAQUS:
        Just export the modal matrix and feed it here.
    """
    # ---- REPLACE THESE ----
    # eigenvalues  = np.load("my_eigenvalues.npy")
    # eigenvectors = np.load("my_eigenvectors.npy")   # shape: (n_dofs, n_modes)
    # n_nodes      = 50
    # node_coords  = np.load("my_node_coords.npy")    # shape: (n_nodes, 3)
    # -----------------------

    # clf = ModeClassifier(eigenvectors, eigenvalues, n_nodes, freq_type='lambda')
    # clf.summary()
    # clf.plot_participation()
    # clf.plot_mode_shape(mode_index=0, node_coords=node_coords)
    pass


# ===========================================================================
if __name__ == "__main__":
    run_demo()
