"""Brief 6 (2026-11-17): what concepts a living layer's activity holds, and
whether they tell experiences apart.

Stage 1 of ``docs/research/2026-11-17_relevant-updates-design.md``. An
instrument on a finished run: no training, plasticity frozen for every
forward, the model checked bit-identical at the end.

Documents are the amendment's unit: ``--read`` windows and then ``--score``
windows of consecutive held-out text. Every other document is used to FIT
(even indices) and the rest to READ (odd indices), so a concept is not a
memory of the passage it was fitted on. Each window is one forward, as a
read is.

A ROW is a living layer's output, token by token, or the EMBEDDING row: the
token embedding alone (no position), the baseline of standing rule 3. A
concept test the embedding passes just as well shows the tokens, not the
trunk.

For each row:

1. Variation. With the mean over tokens removed: its share of the output's
   mean square, its effective rank, and the shares of it that lie between
   documents and between positions in the window.
2. Concepts. K prototypes of the centred outputs, by k-means on cosine, no
   gradient, fitted on the fit half (a sample of ``--fit-tokens`` tokens).
   K at each of ``--k``.
3. What each concept is. For the ``--top-concepts`` most used concepts at
   ``--passages-k``, the passages that drive it most, in a text file.
4. Can a document be told by the concepts it drives? A document's signature
   is how often each concept is driven over its ``--read`` windows. Each of
   its ``--score`` windows is assigned to the document whose signature it
   most resembles (cosine, after the mean signature is removed). Accuracy,
   beside chance and beside the same test with the windows' document labels
   shuffled.
5. Does the error differ by concept? The layer's own error token by token
   (``e = x - y P``), averaged over the tokens that drive each concept: the
   share of the error's variation that lies between concepts, and the
   cosine between different concepts' mean errors. For a living layer this
   is also taken with the EMBEDDING's concepts, as the baseline.
6. The helpful direction (brief 4): minus the gradient of the tail's score
   with respect to the layer's weight. The share of it that lies in the
   span of the concepts the document's read windows drive (the fewest
   concepts that cover half its tokens), beside the share expected if it
   lay anywhere (the span's dimension over the layer's), and beside the
   share in the span of the concepts ANOTHER document drives.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_17/probe_living_concepts.py \\
        --run-dir <run> --docs 48 --out <dir>/concepts.json
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def kmeans_cosine(x: torch.Tensor, k: int, *, iters: int, seed: int) -> torch.Tensor:
    """K unit prototypes of the rows of ``x`` (already centred), by k-means
    on cosine. No gradient. An empty prototype is seeded again from a row."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    xn = torch.nn.functional.normalize(x, dim=1)
    n = xn.shape[0]
    if n < k:
        raise RuntimeError(f"{n} tokens cannot give {k} prototypes")
    protos = xn[torch.randperm(n, generator=gen)[:k].to(xn.device)].clone()
    for _ in range(iters):
        assign = torch.empty(n, dtype=torch.long, device=xn.device)
        for s in range(0, n, 65536):
            assign[s:s + 65536] = (xn[s:s + 65536] @ protos.T).argmax(dim=1)
        sums = torch.zeros_like(protos).index_add_(0, assign, xn)
        counts = torch.bincount(assign, minlength=k)
        empty = (counts == 0).nonzero().flatten()
        if len(empty):
            sums[empty] = xn[torch.randint(0, n, (len(empty),), generator=gen).to(xn.device)]
        protos = torch.nn.functional.normalize(sums, dim=1)
    return protos


def eff_rank_of_cov(cov: torch.Tensor) -> float:
    sv = torch.linalg.eigvalsh(cov.double().cpu()).clamp(min=1e-12)
    p = sv / sv.sum()
    return float(math.exp(float(-(p * torch.log(p)).sum())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--read", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--doc-phase", type=int, default=0)
    ap.add_argument("--k", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--fit-tokens", type=int, default=120_000)
    ap.add_argument("--kmeans-iters", type=int, default=25)
    ap.add_argument("--passages-k", type=int, default=256)
    ap.add_argument("--top-concepts", type=int, default=20)
    ap.add_argument("--passages", type=int, default=10)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remove-position", action="store_true",
                    help="remove each position's own mean before fitting and reading, "
                         "not the one mean over tokens (brief 7, item 4: 40%% of a living "
                         "layer's token variation is position in the window)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_weight_gradient import living_layers, score_and_its_gradient
    from luthi.v2.plasticity import freeze_plasticity
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    text_out = out.with_suffix(".txt")
    if out.exists() or text_out.exists():
        raise SystemExit(f"{out} or {text_out} exists; not overwriting it")
    if args.passages_k not in args.k:
        raise SystemExit("--passages-k must be one of --k")
    device = _device()
    started = time.monotonic()
    run = rebuild_run(args.run_dir, device=device)
    loss_module = run.loss_module.eval()
    model = loss_module.online_encoder
    layers = living_layers(model)
    tokenizer = run.text_ds._tokenizer
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = ae.holdout_documents(windows, n_read=args.read, n_score=args.score,
                                n_docs=args.docs, phase=args.doc_phase)
    fit_docs = [i for i in range(len(docs)) if i % 2 == 0]
    read_docs = [i for i in range(len(docs)) if i % 2 == 1]
    n_r = len(read_docs)
    per = args.read + args.score
    seq = int(windows.shape[1])
    rows = ["embedding"] + [f"layer {i}" for i in range(len(layers))]
    n_rows = len(rows)
    whole_before = {k: v.detach().clone() for k, v in loss_module.state_dict().items()}

    captured: list = [None] * len(layers)

    def hook(index: int):
        def fn(module, inputs, output):
            captured[index] = (inputs[0].detach().reshape(-1, inputs[0].shape[-1]),
                               output.detach().reshape(-1, output.shape[-1]))
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]

    @torch.no_grad()
    def forward(window: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One window. Each row's token outputs, and each living layer's
        token errors ``x - y P``."""
        for layer in layers:
            if not layer._plasticity_frozen:
                raise RuntimeError("a forward with plasticity not frozen")
        model.encode(text_tokens=window.unsqueeze(0), causal=False)
        clear = getattr(model, "clear_forward_cache", None)
        if callable(clear):
            clear()
        outs = [model.embedding(window).detach()]
        errs = []
        for layer, (x, y) in zip(layers, captured):
            outs.append(y)
            errs.append(x - y @ layer.prediction.to(y.dtype))
        return outs, errs

    try:
        with freeze_plasticity(model):
            # ---- pass 1: the fit half ---------------------------------
            fit_total = len(fit_docs) * per * seq
            every = max(1, math.ceil(fit_total / args.fit_tokens))
            pick = torch.arange(args.seed % every, seq, every, device=device)
            kept: list[list[torch.Tensor]] = [[] for _ in rows]
            by_position = None
            fit_windows = 0
            for di in fit_docs:
                for w in docs[di]:
                    outs, _ = forward(w)
                    if by_position is None:
                        by_position = [torch.zeros_like(o, dtype=torch.float64) for o in outs]
                    fit_windows += 1
                    for r in range(n_rows):
                        kept[r].append(outs[r][pick].float().cpu())
                        by_position[r] += outs[r].double()
            fit = [torch.cat(k) for k in kept]
            del kept
            if args.remove_position:
                # One mean a position, [seq, d], from every token of the fit half.
                means = [(b / fit_windows).float().cpu() for b in by_position]
                centred = [f - m[pick.cpu()].repeat(fit_windows, 1) for f, m in zip(fit, means)]
            else:
                means = [f.mean(dim=0) for f in fit]
                centred = [f - m for f, m in zip(fit, means)]
            protos: list[dict[int, torch.Tensor]] = []
            for r in range(n_rows):
                x = centred[r].to(device)
                protos.append({k: kmeans_cosine(x, k, iters=args.kmeans_iters,
                                                seed=args.seed + 1000 * r + k)
                               for k in args.k})
                del x
            fit_tokens = int(fit[0].shape[0])
            del fit, centred
            means = [m.to(device) for m in means]
            print(f"fitted on {fit_tokens} tokens of {len(fit_docs)} documents "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)

            # ---- pass 2: the read half --------------------------------
            n_tok = n_r * per * seq
            assign = {(r, k): torch.empty(n_tok, dtype=torch.int16) for r in range(n_rows)
                      for k in args.k}
            sim = {r: torch.empty(n_tok, dtype=torch.float32) for r in range(n_rows)}
            dims = [int(m.shape[-1]) for m in means]
            s1 = [torch.zeros(d_, dtype=torch.float64, device=device) for d_ in dims]
            s2 = [torch.zeros(d_, d_, dtype=torch.float64, device=device) for d_ in dims]
            by_doc = [torch.zeros(n_r, d_, dtype=torch.float64, device=device) for d_ in dims]
            by_pos = [torch.zeros(seq, d_, dtype=torch.float64, device=device) for d_ in dims]
            # the error, by concept: own concepts and the embedding's concepts
            e_sum = {(li, k, whose): torch.zeros(k, dims[li + 1], dtype=torch.float64, device=device)
                     for li in range(len(layers)) for k in args.k for whose in ("own", "embedding")}
            e_count = {key: torch.zeros(key[1], dtype=torch.float64, device=device) for key in e_sum}
            e_s1 = [torch.zeros(dims[li + 1], dtype=torch.float64, device=device)
                    for li in range(len(layers))]
            e_sq = [0.0] * len(layers)
            at = 0
            for ri, di in enumerate(read_docs):
                for w in docs[di]:
                    outs, errs = forward(w)
                    here: dict = {}
                    for r in range(n_rows):
                        y = outs[r].double()
                        s1[r] += y.sum(dim=0)
                        s2[r] += y.T @ y
                        by_doc[r][ri] += y.sum(dim=0)
                        by_pos[r] += y
                        yc = torch.nn.functional.normalize(outs[r].float() - means[r], dim=1)
                        for k in args.k:
                            sims = yc @ protos[r][k].T
                            best, which = sims.max(dim=1)
                            here[(r, k)] = which
                            assign[(r, k)][at:at + seq] = which.to(torch.int16).cpu()
                            if k == args.passages_k:
                                sim[r][at:at + seq] = best.cpu()
                    for li, e in enumerate(errs):
                        e = e.double()
                        e_s1[li] += e.sum(dim=0)
                        e_sq[li] += float((e * e).sum())
                        ones = torch.ones(seq, dtype=torch.float64, device=device)
                        for k in args.k:
                            for whose, row in (("own", li + 1), ("embedding", 0)):
                                key = (li, k, whose)
                                e_sum[key].index_add_(0, here[(row, k)], e)
                                e_count[key].index_add_(0, here[(row, k)], ones)
                    at += seq
            print(f"read {n_tok} tokens of {n_r} documents "
                  f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    finally:
        for h in handles:
            h.remove()

    # ---- item 6: the helpful direction, by the concepts a document drives
    tok_per_doc = per * seq
    head_tok = args.read * seq
    shift = max(1, n_r // 2)

    def driven(r: int, k: int, ri: int) -> torch.Tensor:
        """The fewest concepts that cover half the tokens of the document's
        read windows."""
        a = assign[(r, k)][ri * tok_per_doc: ri * tok_per_doc + head_tok].long()
        counts = torch.bincount(a, minlength=k).double()
        order = counts.argsort(descending=True)
        cum = counts[order].cumsum(0)
        need = int((cum < 0.5 * counts.sum()).sum()) + 1
        return order[:need]

    def span_share(g: torch.Tensor, directions: torch.Tensor) -> tuple[float, int]:
        q, rdiag = torch.linalg.qr(directions.double().T.cpu())   # [out, m]
        rank = int((rdiag.diagonal().abs() > 1e-8).sum())
        q = q[:, rdiag.diagonal().abs() > 1e-8].to(g.device)
        return float((q.T @ g).pow(2).sum() / g.pow(2).sum()), rank

    helpful: dict = {(li, k): [] for li in range(len(layers)) for k in args.k}
    for ri, di in enumerate(read_docs):
        tail = docs[di][args.read:]
        _, grads = score_and_its_gradient(loss_module, tail)
        for li, g in enumerate(grads):
            g = g.double()
            for k in args.k:
                own = driven(li + 1, k, ri)
                other = driven(li + 1, k, (ri + shift) % n_r)
                share, rank = span_share(g, protos[li + 1][k][own.to(device)])
                share_other, rank_other = span_share(g, protos[li + 1][k][other.to(device)])
                helpful[(li, k)].append({
                    "share": share, "concepts": int(len(own)), "dimension": rank,
                    "expected_if_anywhere": rank / g.shape[0],
                    "share_in_another_documents_concepts": share_other,
                    "another_documents_dimension": rank_other,
                })

    # ---- guard: bit-identical, nothing left on the graph ----------------
    after = loss_module.state_dict()
    changed = [k for k in whole_before if not torch.equal(whole_before[k], after[k])]
    left = [k for k, b in loss_module.named_buffers() if b.requires_grad]
    if changed or left:
        raise RuntimeError(f"the model is not as it was: changed {changed[:5]}, "
                           f"left requiring a gradient {left[:5]}")

    # ---- item 1 ---------------------------------------------------------
    variation = []
    for r in range(n_rows):
        n = float(n_tok)
        mean = s1[r] / n
        cov = s2[r] / n - torch.outer(mean, mean)
        token_var = float(cov.diagonal().sum())
        doc_means = by_doc[r] / tok_per_doc
        pos_means = by_pos[r] / (n_r * per)
        variation.append({
            "row": rows[r],
            "mean_square_of_the_mean": float(mean.pow(2).sum()),
            "token_to_token_variation": token_var,
            "share_of_the_outputs_mean_square": token_var / (token_var + float(mean.pow(2).sum())),
            "effective_rank": eff_rank_of_cov(cov), "dimension": dims[r],
            "share_between_documents": float((doc_means - mean).pow(2).sum(dim=1).mean()) / token_var,
            "share_between_positions": float((pos_means - mean).pow(2).sum(dim=1).mean()) / token_var,
        })

    # ---- item 4 ---------------------------------------------------------
    gen = torch.Generator().manual_seed(args.seed)

    def told_apart(a: torch.Tensor, k: int) -> dict:
        w = a.long().reshape(n_r, per, seq)
        counts = torch.zeros(n_r, per, k, dtype=torch.float64)
        counts.scatter_add_(2, w, torch.ones_like(w, dtype=torch.float64))
        head, held = counts[:, :args.read], counts[:, args.read:]       # [n_r, ., k]

        def accuracy(head_counts: torch.Tensor) -> tuple[float, float, float]:
            sig = head_counts.sum(dim=1)
            sig = sig / sig.sum(dim=1, keepdim=True)
            centre = sig.mean(dim=0, keepdim=True)
            h = held.reshape(n_r * args.score, k)
            h = h / h.sum(dim=1, keepdim=True)
            truth = torch.arange(n_r).repeat_interleave(args.score)
            raw = (torch.nn.functional.normalize(h, dim=1)
                   @ torch.nn.functional.normalize(sig, dim=1).T).argmax(dim=1)
            cen = (torch.nn.functional.normalize(h - centre, dim=1)
                   @ torch.nn.functional.normalize(sig - centre, dim=1).T).argmax(dim=1)
            pooled = held.sum(dim=1)
            pooled = pooled / pooled.sum(dim=1, keepdim=True)
            doc = (torch.nn.functional.normalize(pooled - centre, dim=1)
                   @ torch.nn.functional.normalize(sig - centre, dim=1).T).argmax(dim=1)
            return (float((cen == truth).double().mean()), float((raw == truth).double().mean()),
                    float((doc == torch.arange(n_r)).double().mean()))

        acc, acc_raw, acc_doc = accuracy(head)
        shuffled = []
        flat = head.reshape(n_r * args.read, k)
        for _ in range(args.shuffles):
            perm = torch.randperm(n_r * args.read, generator=gen)
            shuffled.append(accuracy(flat[perm].reshape(n_r, args.read, k))[0])
        used = counts.sum(dim=(0, 1))
        p = used / used.sum()
        return {
            "by_window": acc, "by_window_mean_signature_not_removed": acc_raw,
            "by_document_all_held_windows_together": acc_doc,
            "chance": 1.0 / n_r,
            "labels_shuffled": {"mean": statistics.fmean(shuffled), "max": max(shuffled),
                                "shuffles": len(shuffled)},
            "windows": n_r * args.score, "documents": n_r,
            "concepts_used": int((used > 0).sum()),
            "effective_number_of_concepts": float(math.exp(float(-(p[p > 0] * p[p > 0].log()).sum()))),
        }

    apart = {(r, k): told_apart(assign[(r, k)], k) for r in range(n_rows) for k in args.k}

    # ---- item 5 ---------------------------------------------------------
    def error_by_concept(li: int, k: int, whose: str) -> dict:
        key = (li, k, whose)
        counts = e_count[key]
        n = float(counts.sum())
        mean = e_s1[li] / n
        total = e_sq[li] - n * float(mean.pow(2).sum())
        live = counts > 0
        mu = e_sum[key][live] / counts[live].unsqueeze(1)
        between = float((counts[live] * (mu - mean).pow(2).sum(dim=1)).sum())
        cen = torch.nn.functional.normalize(mu - mean, dim=1)
        raw = torch.nn.functional.normalize(mu, dim=1)
        iu = torch.triu_indices(mu.shape[0], mu.shape[0], offset=1)
        c_raw = (raw @ raw.T)[iu[0], iu[1]]
        c_cen = (cen @ cen.T)[iu[0], iu[1]]
        return {
            "share_of_the_errors_variation_between_concepts": between / total,
            "share_expected_by_chance": (int(live.sum()) - 1) / (n - 1),
            "the_same_for_every_token_share_of_the_errors_mean_square":
                n * float(mean.pow(2).sum()) / e_sq[li],
            "cosine_between_concepts_mean_errors": {
                "median": float(c_raw.median()), "min": float(c_raw.min()), "max": float(c_raw.max())},
            "cosine_between_concepts_mean_errors_common_part_removed": {
                "median": float(c_cen.median()), "min": float(c_cen.min()), "max": float(c_cen.max())},
            "concepts_driven": int(live.sum()),
        }

    errors = {(li, k, whose): error_by_concept(li, k, whose) for (li, k, whose) in e_sum}

    def med(values: list[float]) -> float:
        return statistics.median(values)

    result_rows = []
    for r in range(n_rows):
        row = {"row": rows[r], "variation": variation[r], "by_k": {}}
        for k in args.k:
            entry = {"told_apart": apart[(r, k)]}
            if r > 0:
                li = r - 1
                entry["error_by_concept"] = {"own_concepts": errors[(li, k, "own")],
                                             "embedding_concepts": errors[(li, k, "embedding")]}
                h = helpful[(li, k)]
                entry["helpful_direction"] = {
                    "share_median": med([x["share"] for x in h]),
                    "expected_if_anywhere_median": med([x["expected_if_anywhere"] for x in h]),
                    "share_in_another_documents_concepts_median":
                        med([x["share_in_another_documents_concepts"] for x in h]),
                    "concepts_median": med([x["concepts"] for x in h]),
                    "dimension_median": med([x["dimension"] for x in h]),
                    "documents_where_own_exceeds_anothers": sum(
                        x["share"] > x["share_in_another_documents_concepts"] for x in h),
                    "documents": len(h), "by_document": h,
                }
            row["by_k"][str(k)] = entry
        result_rows.append(row)

    # ---- item 3: what each concept is, as text --------------------------
    lines = [f"Concepts of {run.result['arm']} seed {run.result['seed']} "
             f"({Path(args.run_dir).name}), K = {args.passages_k}, document set {args.doc_phase}.",
             "For each row, its most used concepts, and for each the passages that drive it",
             "most. The driving token is in [[double brackets]]. Fitted on half the",
             "documents; these passages are from the other half.", ""]
    k = args.passages_k
    read_tokens = torch.stack([docs[di] for di in read_docs]).reshape(-1).cpu()
    win_of = torch.arange(n_tok) // seq

    def passage(t: int) -> str:
        w = t // seq
        lo, hi = max(w * seq, t - 10), min((w + 1) * seq, t + 5)
        before = tokenizer.decode(read_tokens[lo:t].tolist())
        token = tokenizer.decode(read_tokens[t:t + 1].tolist())
        after_ = tokenizer.decode(read_tokens[t + 1:hi].tolist())
        return (before + "[[" + token + "]]" + after_).replace("\n", " ").replace("\r", " ")

    concepts_out = []
    for r in range(n_rows):
        a = assign[(r, k)].long()
        used = torch.bincount(a, minlength=k)
        lines += ["=" * 72, rows[r].upper(), "=" * 72]
        for c in used.argsort(descending=True)[:args.top_concepts].tolist():
            idx = (a == c).nonzero().flatten()
            order = sim[r][idx].argsort(descending=True)
            chosen, seen = [], set()
            for j in order.tolist():
                t = int(idx[j])
                if int(win_of[t]) in seen:
                    continue
                seen.add(int(win_of[t]))
                chosen.append(t)
                if len(chosen) == args.passages:
                    break
            docs_using = len(set((idx // tok_per_doc).tolist()))
            share = float(used[c]) / n_tok
            lines.append(f"concept {c}: {share:.2%} of tokens, driven in {docs_using} of {n_r} documents")
            passages = [passage(t) for t in chosen]
            lines += [f"    {p}" for p in passages] + [""]
            concepts_out.append({"row": rows[r], "concept": c, "share_of_tokens": share,
                                 "documents_driving_it": docs_using, "passages": passages})

    report = {
        "what": "brief 6: concepts of each living layer's token outputs, and whether they "
                "tell documents apart",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "documents": {"n_docs": len(docs), "n_read": args.read, "n_score": args.score,
                      "phase": args.doc_phase, "fit_documents": fit_docs,
                      "read_documents": read_docs, "seq_len": seq,
                      "holdout_windows": int(windows.shape[0])},
        "settings": {"k": args.k, "fit_tokens": fit_tokens, "fit_every_nth_token": every,
                     "kmeans_iters": args.kmeans_iters, "seed": args.seed,
                     "mean_removed": ("each position's own" if args.remove_position
                                      else "one mean over tokens"),
                     "read_tokens": n_tok, "one_window_a_forward": True,
                     "passages_k": args.passages_k, "device": str(device)},
        "guards": {"plasticity_frozen_for_every_forward": True,
                   "bit_identical_afterwards": True, "buffers_left_requiring_grad": 0},
        "rows": result_rows,
        "concepts_at_passages_k": concepts_out,
        "minutes": (time.monotonic() - started) / 60,
        "reconstruction": reconstruction(run.result["config"]),
    }

    print(f"{run.result['arm']} seed {run.result['seed']}; {n_r} documents read; chance {1 / n_r:.3f}")
    print(f"{'row':<10} {'var share':>9} {'eff rank':>8} {'btw docs':>8} {'btw pos':>8}")
    for v in variation:
        print(f"{v['row']:<10} {v['share_of_the_outputs_mean_square']:>9.3f} "
              f"{v['effective_rank']:>8.1f} {v['share_between_documents']:>8.3f} "
              f"{v['share_between_positions']:>8.3f}")
    for k_ in args.k:
        print(f"K = {k_}")
        print(f"  {'row':<10} {'window acc':>10} {'shuffled':>8} {'doc acc':>8} "
              f"{'err btw':>8} {'err emb':>8} {'err cos':>8} {'help':>7} {'anywhere':>8} {'another':>8}")
        for r, row in enumerate(result_rows):
            e = row["by_k"][str(k_)]
            t = e["told_apart"]
            line = (f"  {row['row']:<10} {t['by_window']:>10.3f} {t['labels_shuffled']['mean']:>8.3f} "
                    f"{t['by_document_all_held_windows_together']:>8.3f}")
            if r > 0:
                eo, ee = e["error_by_concept"]["own_concepts"], e["error_by_concept"]["embedding_concepts"]
                h = e["helpful_direction"]
                line += (f" {eo['share_of_the_errors_variation_between_concepts']:>8.3f}"
                         f" {ee['share_of_the_errors_variation_between_concepts']:>8.3f}"
                         f" {eo['cosine_between_concepts_mean_errors']['median']:>+8.2f}"
                         f" {h['share_median']:>7.3f} {h['expected_if_anywhere_median']:>8.3f}"
                         f" {h['share_in_another_documents_concepts_median']:>8.3f}")
            print(line)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps(report), encoding="utf-8")
    text_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} and {text_out.name} ({report['minutes']:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
