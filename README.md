# crosslingual-steering

Does an English→German activation steering vector generalize across text
domains, or does it overfit to the domain it was built from — and how
sensitive is it to the strength of the push?

A small, targeted experiment testing the robustness of DiffMean/ActAdd-style
steering vectors, extending a question left open by Turner et al. (2023,
*Activation Addition*) and Gurgurov et al. (2026, *CLaS-Bench*), both of
which build and evaluate steering vectors on a single distribution of text.

## Motivation

Activation steering — adding a direction vector into a model's residual
stream at inference time to control its output — is increasingly proposed as
a lightweight alternative to fine-tuning (Turner et al., 2023) and has been
benchmarked at scale for cross-lingual control by CLaS-Bench (Gurgurov et
al., 2026), which shows DiffMean-style steering outperforming prompting and
other representation-based methods across 32 languages.

CLaS-Bench's own ablations show that steering effectiveness depends sharply
on *layer* and *coefficient strength*. What neither that work nor the
original ActAdd paper tests is whether a steering vector built from one
*domain* of text (e.g. plain factual sentences) still works when applied to
a different domain (e.g. opinion or creative-writing prompts) — a direct,
practical instance of the "robustness under distribution shift" question.

## Method

1. **Model:** `meta-llama/Llama-3.2-1B-Instruct`, run via plain HuggingFace
   `transformers` with a manual `register_forward_hook` on one decoder
   layer (≈60% depth into the network) — no fine-tuning, no gradient
   descent.
2. **Steering vector:** built ActAdd-style from 10 short English–German
   contrastive sentence pairs. The mean last-token residual-stream activation
   for the German sentences minus the mean for the English sentences gives
   the "German-ness" direction; this is L2-normalized (matching CLaS-Bench's
   own DiffMean formulation) before being scaled by a coefficient and added
   back into the residual stream during generation.
3. **Evaluation questions:** the English subset of CLaS-Bench's own 70-question
   benchmark (`DGurgurov/CLaS-Bench` on Hugging Face), auto-bucketed into
   domains (knowledge, creative, opinion, reasoning, writing) with a keyword
   heuristic, since the public dataset doesn't ship the paper's manual domain
   labels.
4. **Metrics:** Language Forcing Success (LFS, via `langdetect`) and a
   perplexity-based relevance/coherence proxy, combined into a harmonic-mean
   Language Steering Score (LSS) — deliberately mirroring CLaS-Bench's own
   LFS/OR/LSS design, though using a lightweight self-perplexity proxy in
   place of their paid LLM-judge relevance score.
5. **Design:** for three domains (knowledge, creative, opinion) and several
   coefficient strengths, generate steered completions and measure LFS,
   relevance, and LSS, comparing against an unsteered (coeff=0) baseline.

## Results

Across two runs (N=3 and N=5 questions per domain), the same qualitative
pattern held: steering had **no effect below a coefficient threshold**
(~2.0–2.5 in this setup), **peaked sharply around one specific coefficient**
(~2.8) across all three tested domains, and **collapsed into incoherent
output well above that threshold** — again, in every domain tested.

| Domain    | coeff=0 (baseline) | coeff≈2.8 (peak) | coeff≈5.0 (overshoot) |
|-----------|---------------------|-------------------|--------------------------|
| knowledge | LSS ≈ 0 (no steering) | LSS ≈ 0.68 | LSS ≈ 0.17 |
| creative  | LSS ≈ 0 | LSS ≈ 0.82 | LSS ≈ 0.00 |
| opinion   | LSS ≈ 0 | LSS ≈ 0.59 | LSS ≈ 0.26 |

The relative ranking of domains at peak coefficient was **not stable**
across the two runs (small N — 3 to 5 questions per domain), so we do not
claim one domain is more robust to steering than another. What *was*
consistent across runs: a narrow, sharply-bounded effective coefficient
window rather than a smooth degradation curve, and that window appeared to
sit in roughly the same place regardless of question domain.

**Interpretation:** for this steering vector, robustness under distribution
shift looks more sensitive to precise coefficient calibration than to the
semantic domain of the input text. Overshooting the effective window doesn't
degrade output gracefully — it breaks coherence sharply, in every domain
alike.

Sample outputs (full text in `domain_shift_results.csv`) show real German
vocabulary and grammar mixed with residual English at the peak coefficient
(e.g. *"Zeitmanagers helfen, gute Zeit- und project-Organisation..."*),
confirming genuine partial language-steering rather than a language-detector
artifact.

## Limitations

- Small sample size (3–5 questions per domain) — domain rankings are noisy;
  the coefficient-sensitivity finding is more robust than any domain-specific
  ranking.
- Domain labels are auto-tagged via keyword heuristics, not the original
  paper's manual labels (not publicly released with the dataset).
- Relevance/coherence is scored via self-perplexity, a lightweight proxy for
  CLaS-Bench's LLM-as-judge relevance score.
- Single model (Llama-3.2-1B-Instruct), single layer, single language pair
  (English→German) — not a claim about steering in general, just this
  specific configuration.
- `langdetect` occasionally misclassifies short or heavily code-mixed text;
  spot-checked against raw generations for the reported coefficients.

## Repository structure

```
crosslingual-steering/
├── README.md
├── run_experiment.py        # end-to-end script: load model, build vector,
│                             # generate, score, plot
├── domain_shift_results.csv # per-generation results (question, output,
│                             # LFS, relevance, LSS)
└── domain_shift_lss.png     # LSS vs. coefficient, one line per domain
```

## How to run

```bash
pip install transformers langdetect datasets pandas matplotlib torch huggingface_hub
hf auth login   # needed for gated Llama weights
python run_experiment.py
```

Requires accepting Meta's license for `meta-llama/Llama-3.2-1B-Instruct` on
Hugging Face before first run.

## References

- Turner, A. M., et al. (2023). *Activation Addition: Steering Language
  Models Without Optimization.* arXiv:2308.10248.
- Gurgurov, D., et al. (2026). *CLaS-Bench: A Cross-Lingual Alignment and
  Steering Benchmark.* Findings of ACL 2026.
- Gurgurov, D., et al. (2025). *Language Arithmetics: Towards Systematic
  Language Neuron Identification and Manipulation.* IJCNLP-AACL 2025.
