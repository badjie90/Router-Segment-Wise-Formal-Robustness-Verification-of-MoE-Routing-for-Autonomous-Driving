# Router-Segment-Wise-Formal-Robustness-Verification-of-MoE-Routing-for-Autonomous-Driving
Router Segment-Wise Formal Robustness Verification of Mixture-of-Experts Routing for Autonomous Driving




# BDD100K MoE Gate: Segment-Wise Adversarial Evaluation and Formal Verification

This project analyzes the trained condition-specialized BDD100K Mixture-of-
Experts (MoE) gate at both its intermediate representations and final routing
decision. It combines empirical black-box attacks with sound bound-propagation
certificates and downstream task evaluation.

The source MoE is loaded from the existing baseline project; it is not retrained
or silently replaced:

```text
../Baseline/BDD100k/
├── scripts/bdd100k_moe_train.py
├── data/metadata_files/metadata/test_fixed.json
└── train_models/moe_stage3/
    ├── config.json
    └── checkpoints/best.pt
```

## Scientific question

Previous experiments identified the gate as the most vulnerable MoE attack
surface under HopSkipJump (HSJ), Square Attack, and transfer-based PGD. This
project asks where that vulnerability emerges and whether robustness can be
certified before the final gate output.

The effective gate is:

```text
image → ConvNeXt-V2 backbone → pooled feature (768)
      → LayerNorm → Linear(768,256) → GELU → Dropout(eval=identity)
      → Linear(256,3) → routing logits → softmax expert weights
```

Verification cut points are:

1. `backbone_features`
2. `router_normalized`
3. `router_hidden_affine`
4. `router_hidden_gelu`
5. `router_logits`

## Verification methodology

The primary formal backend is optimized CROWN bound propagation through
`auto_LiRPA`. For an input image `x` and an L-infinity pixel-space threat set
`||x' - x||∞ ≤ ε`, clipped to `[0,1]`, it computes sound lower/upper bounds for
each cut-point coordinate and for pairwise routing margins.

For each sample, final routing is certified stable when the lower bound of

```text
logit(clean_top1) - logit(other)
```

is strictly positive for every competing expert. Intermediate robustness is
reported without inventing a discrete “intermediate class”: coordinate interval
width, certified L-infinity/L2 representation radius, relative interval width,
and an upper bound on local amplification. These are valid properties of a
continuous hidden representation.

When optimized CROWN is inconclusive, Alpha-Beta-CROWN complete verification
with branch-and-bound/BICCOS is the intended escalation tier. Full ConvNeXt
complete verification is expensive, so undecided results must remain
`unknown`—never relabeled robust.

## Empirical attacks

- **HopSkipJump:** decision-based attack against the clean top-1 route.
- **Square Attack:** score-based L-infinity attack against the route margin.
- **Transfer PGD:** trains a small convolutional surrogate on target-gate
  pseudo-labels, then performs multi-restart PGD against the surrogate.

Attacks operate on raw pixel tensors in `[0,1]`; ImageNet normalization is inside
the model wrapper. This keeps epsilon scientifically interpretable.

## Metrics

Formal metrics:

- certified routing accuracy/rate at each epsilon;
- verified-safe, falsified-by-attack, and unknown counts;
- certified pairwise route-margin lower bound;
- coordinate interval mean/max width per segment;
- certified representation L-infinity radius;
- certified representation L2 radius upper bound;
- certified local amplification upper bound;
- verification runtime and error/unknown rate.

The formal analysis uses a router-feature L-infinity threat model. For each
image, the frozen ConvNeXtV2 backbone produces a 768-dimensional feature vector;
optimized CROWN then certifies the router and every intermediate router segment
for radii equal to 2%, 4%, and 8% of that clean vector's standard deviation.
This is deliberately reported separately from the empirical pixel-space threat
model: the feature radii are not claimed to be equivalent to pixel radii of
2/255, 4/255, or 8/255. End-to-end pixel attacks still include the backbone.

Empirical metrics:

- attack success rate and robust route accuracy;
- top-1 route agreement and Jensen-Shannon divergence;
- clean/adversarial route margin;
- L-infinity and L2 perturbation norms;
- queries or explicitly labeled query upper bounds, and runtime;
- segment cosine similarity and relative L2 drift;
- fused MoE macro/micro AUROC, average precision, F1, balanced accuracy, and ECE;
- per-object metrics and clean-to-adversarial degradation.

Confidence intervals use paired bootstrap resampling over images. Attack and
verification results are always reported separately.

## Installation

```bash
cd BDD100K-Gate-Segment-Verification
conda create --name bdd-gate-verifier python=3.11 pip -y
conda activate bdd-gate-verifier
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The current Alpha-Beta-CROWN/auto_LiRPA release requires Python 3.11, NumPy 2,
and a CUDA-compatible PyTorch build. Clone it recursively and install its bound
propagation library into the same environment:

```bash
cd /mnt/nvme1n1/bbadjie
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
# If an earlier clone was interrupted, repair it instead:
cd alpha-beta-CROWN
git submodule update --init --recursive
python -m pip install -e auto_LiRPA
python -m pip install -e /mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100K-Gate-Segment-Verification
```

Do not use the obsolete `complete_verifier/environment.yaml` command with the
current repository revision; that file has been replaced by `pyproject.toml`.

## Configuration

Copy and edit the example:

```bash
cp configs/bdd100k_gate.yaml.example configs/bdd100k_gate.yaml
```

All paths may be relative to this project. Verify that image paths stored in
`test_fixed.json` exist on the current machine.

`project.require_cuda` defaults to `true`. Setup, attacks, and verification fail
immediately if the configured CUDA device is unavailable; they never silently
fall back to CPU. Change this only for short debugging runs when CPU use is
explicitly permitted by your computing environment.

## Run order

### 1. Validate the source model and data

```bash
python scripts/check_setup.py --config configs/bdd100k_gate.yaml
```

### 2. Run clean and adversarial evaluation

Start with a smoke test:

```bash
python scripts/run_attacks.py --config configs/bdd100k_gate.yaml \
  --attacks hsj square transfer_pgd --max-samples 10
```

Then increase `max_samples`, query budgets, and PGD restarts for the final study.

### 3. Run sound segment-wise verification

```bash
python scripts/run_verification.py --config configs/bdd100k_gate.yaml \
  --max-samples 10
```

Run epsilon values separately if GPU memory is limited. Verification records
`unknown` on timeout or unsupported operators rather than falling back to an
unsound estimate.

### 4. Aggregate metrics and plots

```bash
python scripts/make_report.py --config configs/bdd100k_gate.yaml
```

## Output structure

```text
outputs/
├── attacks/
│   ├── sample_metrics.csv
│   ├── segment_metrics.csv
│   └── predictions.npz
├── verification/
│   ├── certificates.csv
│   └── segment_bounds.csv
└── report/
    ├── summary.json
    ├── aggregate_metrics.csv
    └── plots/
        ├── certified_rate_vs_epsilon.pdf
        ├── attack_success_vs_epsilon.pdf
        ├── segment_bound_width.pdf
        ├── segment_empirical_drift.pdf
        ├── route_margin_clean_adv.pdf
        └── verification_outcomes.pdf
```

## Interpretation rules

- `certified`: the verifier proved the property over the complete threat set.
- `falsified`: an attack found a counterexample inside the threat set.
- `unknown`: neither proof nor counterexample was obtained within resources.
- Attack failure is not a certificate.
- A narrow hidden interval is not final-route certification unless all pairwise
  routing margins are also proved positive.
- Conditional suffix verification must be labeled conditional; only bounds
  originating from the pixel threat set are end-to-end certificates.
- Report the number of verified, falsified, unknown, timed-out, and errored
  samples. Never drop difficult samples from the denominator.

## Reproducibility

Record the baseline commit/checkpoint SHA-256, fixed test metadata, sample IDs,
epsilon, norm, verifier method, optimization iterations, timeout, attack budgets,
surrogate architecture/training seed, package versions, GPU, and CUDA version.
Use the same sample subset for attacks and certificates.

## Current implementation status

The empirical pipeline, segmentation adapters, surrogate model, metrics, bound
driver, and report generator are included. Formal verification requires a
working `auto_LiRPA`/Alpha-Beta-CROWN environment. ConvNeXt operator support and
memory use must be validated with `check_setup.py` before a large run. Unsupported
operations are a reported limitation, not grounds for an approximate certificate.

## Citation

Publications should cite BDD100K, ConvNeXt-V2/timm, HopSkipJump, Square Attack,
PGD, auto_LiRPA, α-CROWN, β-CROWN, and BICCOS when those components are used.
Also cite the original baseline MoE repository/checkpoint used in the study.





# INSTRUCTIONS to RUN

#### Activate this environment ----- conda activate bdd-gate-verifier

#### The go to this directory ---- /mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100K-Gate-Segment-Verification



1. RUN  the empirical attacks:

python scripts/run_attacks.py \
  --config configs/bdd100k_gate.yaml \
  --attacks hsj square transfer_pgd \
  --max-samples 10

---max_samples 10 means only 10 samples are selected for testing. Remove it when you want to run the full attacks.


---- If you want, you can run the attacks one at a time using 
python scripts/run_attacks.py \
  --config configs/bdd100k_gate.yaml \
  --attacks hsj



2. Then run formal segment verification:


python scripts/run_verification.py \
  --config configs/bdd100k_gate.yaml



3. Finally, generate the consolidated metrics and plots:


1. python scripts/make_report.py \
  --config configs/bdd100k_gate.yaml
