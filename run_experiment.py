"""
run_experiment.py
==================

Domain-Shift Robustness of Activation Steering (English -> German)

Does a DiffMean/ActAdd-style steering vector built from ONE domain of
text still work when applied to prompts from a DIFFERENT domain?

Rewritten to use plain HuggingFace `transformers` with a manual forward
hook instead of `transformer_lens` -- transformer_lens's own loading
pipeline was causing repeated, untraceable process deaths on this
machine, likely in its internal conversion step. Plain `transformers`
is the simpler, more standard path for this and has none of that
extra conversion overhead.
"""

import os
os.environ.setdefault("HF_HUB_CACHE", r"E:\huggingface_cache\hub")

import sys
import time
import traceback

import torch
import numpy as np
import pandas as pd

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER_FRACTION = 0.6
COEFFS_FRACTIONS = [0.4, 0.55, 0.7]  # focus around where steering actually kicks in
N_PER_DOMAIN = 5
MAX_NEW_TOKENS = 40
TRAIN_DOMAIN = "knowledge"
TEST_DOMAINS = ["knowledge", "creative", "opinion"]

RESULTS_CSV = "domain_shift_results.csv"
RESULTS_PNG = "domain_shift_lss.png"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Starting. Importing heavy libraries (transformers, langdetect, datasets)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from langdetect import detect, DetectorFactory
    from datasets import load_dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DetectorFactory.seed = 0
    device = "cpu"
    log(f"Using device={device}")

    # ---- Step 1: load model (plain HF, no transformer_lens) ----
    log(f"Loading tokenizer '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    log(f"Loading model '{MODEL_NAME}' (this can take 1-3 minutes on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    log("Model object created.")

    decoder_layers = model.model.layers  # works for Llama-family models
    n_layers = len(decoder_layers)
    log(f"Model loaded. n_layers={n_layers}, hidden_size={model.config.hidden_size}")
    LAYER = max(0, min(n_layers - 1, int(round(n_layers * LAYER_FRACTION)) - 1))
    log(f"Auto-selected LAYER={LAYER} (fraction {LAYER_FRACTION} of {n_layers} layers)")

    def to_tokens(text):
        return tokenizer(text, return_tensors="pt").to(device)

    def to_string(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    # ---- Step 2: load and bucket CLaS-Bench questions ----
    log("Downloading CLaS-Bench from Hugging Face (single combined table)...")
    ds = load_dataset("DGurgurov/CLaS-Bench", split="train")
    full_df = ds.to_pandas()
    log(f"Loaded {len(full_df)} total rows across all languages.")
    en_df = full_df[full_df["language_code"] == "en"][["question_id", "question"]].sort_values("question_id")
    en_df = en_df.drop_duplicates(subset="question_id")
    log(f"Filtered to {len(en_df)} English questions.")
    if len(en_df) == 0:
        raise ValueError("No English rows found — check the language_code values: "
                          f"{full_df['language_code'].unique()[:10]}")

    def guess_domain(q: str) -> str:
        ql = q.lower()
        if ql.startswith(("how many", "how much")) or "step-by-step" in ql or "step by step" in ql:
            return "reasoning"
        if any(k in ql for k in ["imagine", "pretend", "as a", "as if", "medieval", "superhero",
                                  "time traveler", "colonist", "chef", "climber", "commentator"]):
            return "creative"
        if any(k in ql for k in ["moral", "should we", "do we have", "obligation", "opinion",
                                  "prefer", "important to"]):
            return "opinion"
        if any(k in ql for k in ["write a", "draft", "email", "letter", "script", "blog post",
                                  "review", "outline"]):
            return "writing"
        return "knowledge"

    en_df["domain"] = en_df["question"].apply(guess_domain)
    log("Domain counts (auto-tagged, sanity check these):")
    print(en_df["domain"].value_counts())

    # ---- Step 3: build steering vector via a temporary capture hook ----
    log(f"Building English->German steering vector at layer {LAYER}...")

    contrast_pairs = [
        ("I love talking about this.", "Ich liebe es, darüber zu sprechen."),
        ("This is a good idea.", "Das ist eine gute Idee."),
        ("Let me explain how it works.", "Lass mich erklären, wie es funktioniert."),
        ("The weather is nice today.", "Das Wetter ist heute schön."),
        ("I am not sure about that.", "Ich bin mir da nicht sicher."),
        ("Thank you for your help.", "Danke für deine Hilfe."),
        ("Here are a few tips.", "Hier sind ein paar Tipps."),
        ("This is an important question.", "Das ist eine wichtige Frage."),
        ("I will try my best.", "Ich werde mein Bestes versuchen."),
        ("Let's think about this carefully.", "Lass uns das sorgfältig überlegen."),
    ]
    english_sents = [p[0] for p in contrast_pairs]
    german_sents = [p[1] for p in contrast_pairs]

    captured = {}

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["last_token"] = hidden[0, -1, :].detach().clone()
        return output

    def get_mean_resid(sentences):
        handle = decoder_layers[LAYER].register_forward_hook(capture_hook)
        vecs = []
        try:
            for s in sentences:
                enc = to_tokens(s)
                with torch.no_grad():
                    model(**enc)
                vecs.append(captured["last_token"])
        finally:
            handle.remove()
        return torch.stack(vecs).mean(dim=0)

    h_de = get_mean_resid(german_sents)
    h_en = get_mean_resid(english_sents)
    raw_diff = h_de - h_en
    diff_norm = raw_diff.norm().item()
    steer_vec = (raw_diff / diff_norm).to(device)
    log(f"Steering vector built. Raw diff norm = {diff_norm:.3f} (now normalized to unit length).")

    # Calibration
    handle = decoder_layers[LAYER].register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**to_tokens("The weather today is quite pleasant."))
    handle.remove()
    typical_resid_norm = captured["last_token"].norm().item()
    log(f"Typical residual-stream norm at layer {LAYER}: {typical_resid_norm:.3f}")
    COEFFS = [round(f * typical_resid_norm, 1) for f in COEFFS_FRACTIONS]
    log(f"Auto-selected COEFFS = {COEFFS} (fractions {COEFFS_FRACTIONS} of typical norm)")

    # ---- Step 4: generation + steering hook ----
    def make_steer_hook(vec, coeff):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden + coeff * vec
                return (hidden,) + output[1:]
            else:
                return output + coeff * vec
        return hook_fn

    def generate(prompt, coeff=0.0):
        handle = None
        if coeff != 0.0:
            handle = decoder_layers[LAYER].register_forward_hook(make_steer_hook(steer_vec, coeff))
        try:
            enc = to_tokens(prompt)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0][enc["input_ids"].shape[1]:]
            return to_string(new_ids)
        finally:
            if handle is not None:
                handle.remove()

    def is_german(text):
        text = text.strip()
        if len(text) < 3:
            return False
        try:
            return detect(text) == "de"
        except Exception:
            return False

    def self_perplexity(text):
        if len(text.strip()) < 3:
            return float("inf")
        enc = to_tokens(text)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        return torch.exp(out.loss).item()

    def relevance_from_ppl(ppl, ppl_lo=10.0, ppl_hi=200.0):
        if not np.isfinite(ppl):
            return 0.0
        score = 1.0 - (ppl - ppl_lo) / (ppl_hi - ppl_lo)
        return float(np.clip(score, 0.0, 1.0))

    def harmonic_mean(a, b, eps=1e-8):
        if a + b < eps:
            return 0.0
        return 2 * a * b / (a + b)

    # ---- Step 5: run the experiment ----
    results = []
    total_calls = 0
    for domain in TEST_DOMAINS:
        pool = en_df[en_df.domain == domain]["question"].tolist()
        if not pool:
            log(f"WARNING: no questions found for domain '{domain}', skipping.")
            continue
        sample = pool[:N_PER_DOMAIN]
        total_calls += len(sample) * (len(COEFFS) + 1)
    log(f"Will run {total_calls} total generations. This may take a while on CPU.")

    call_idx = 0
    for domain in TEST_DOMAINS:
        pool = en_df[en_df.domain == domain]["question"].tolist()
        if not pool:
            continue
        sample = pool[:N_PER_DOMAIN]
        log(f"--- Domain: {domain} ({len(sample)} questions) ---")

        for coeff in COEFFS + [0.0]:
            for q in sample:
                call_idx += 1
                log(f"[{call_idx}/{total_calls}] domain={domain} coeff={coeff} q='{q[:50]}...'")
                out = generate(q, coeff=coeff)
                lfs = 1.0 if is_german(out) else 0.0
                ppl = self_perplexity(out)
                rel = relevance_from_ppl(ppl)
                lss = harmonic_mean(lfs, rel)
                results.append({
                    "domain": domain, "coeff": coeff, "question": q, "output": out,
                    "lfs": lfs, "relevance": rel, "lss": lss,
                })

    # ---- Step 6: save results ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    log(f"Saved results to {RESULTS_CSV}")
    print(results_df.groupby(["domain", "coeff"])[["lfs", "relevance", "lss"]].mean())

    # ---- Step 7: plot ----
    summary = results_df[results_df.coeff != 0.0].groupby(["domain", "coeff"])["lss"].mean().reset_index()
    plt.figure(figsize=(7, 5))
    for domain in TEST_DOMAINS:
        sub = summary[summary.domain == domain].sort_values("coeff")
        if len(sub):
            plt.plot(sub["coeff"], sub["lss"], marker="o", label=domain)
    plt.xlabel("Steering coefficient")
    plt.ylabel("Language Steering Score (LSS)")
    plt.title(f"English->German steering: in-domain ({TRAIN_DOMAIN}) vs out-of-domain\n"
              f"{MODEL_NAME}, layer {LAYER}")
    plt.legend(title="Question domain")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=150)
    log(f"Saved plot to {RESULTS_PNG}")
    log("DONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n\n===== SCRIPT FAILED — full traceback below =====", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)