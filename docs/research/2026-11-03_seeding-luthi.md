# Seeding Luthi: what the research says and a path to the best start — 2026-11-03

- **Asked by:** Brian, 2026-11-03: "Maybe the real breakthrough is better seeding rather than building a finished model from scratch." Then: "Please do a deeper dive into research that is even peripherally related to this concept and come up with a more robust path for this idea. We need to give Luthi the best possible start."
- **Written by:** Claude (Opus 5.5).
- **Status:** a proposal. It is not a registration and changes nothing in E1–E5.
- **Governed by:** Brian's ruling of the same day (`DECISIONS.md`, 2026-11-03). Before birth we shape; after it, Luthi decides. Every pre-birth choice is recorded, never hidden, and not explained before it matters or before Luthi asks.

---

## 1. What "the best start" has to mean

A start can be judged on five things:

1. **Healthy representations from step zero:** no collapse, and spread across many directions. LeJEPA proves an isotropic Gaussian is the optimal embedding distribution for a JEPA, which is what SIGReg pushes toward [21].
2. **A living channel that can carry what it is for** (standing rule 1). A seed cannot fix the write rule's capacity; E3 owns that. It must not make that capacity harder to use.
3. **A home worth returning to.** See section 2. This is the property most specific to Luthi.
4. **Less training.** It is shorter, cheaper, and closer to Brian's unease about what training does to models.
5. **Plasticity kept.** Luthi learns for life, so a start that is too settled is a bad start (section 3.7).

## 2. The finding that makes this urgent: Luthi's home is a random matrix

From the code (`luthi/v2/living_layer_pc.py`, `luthi/v2/pc_ops.py`, `luthi/v2/hybrid_block_pc.py`):

- **The home starts as the random draw.** The living FFN's `weight` is drawn at random: Kaiming-uniform, or μPC normal when μPC is on. `set_point`, the home it is pulled back to, starts as an exact copy of that draw.
- **The weight is pulled home fast.** Every forward, homeostasis moves `weight` toward `set_point` at `homeostatic_decay = 1e-3`. The code's own comment puts the time constant at ~1000 forwards.
- **The home barely moves.** `set_point` follows the weight at `set_point_adapt_rate = 1e-6` per forward, which needs on the order of a million forwards to close a gap. A 54,000-step family (the 768x8 run) moves its home only a few percent of the way. Sanctuary's NREM consolidation moves it faster during life (1% of the gap per NREM cycle), but only after birth.
- **No gradient ever touches it.** The living FFN receives no gradient (CLAUDE.md, "Do not reinvent" 1). Training therefore never improves the home. Only PC writes move the weight, and homeostasis keeps pulling it back.

**So in every run so far, the state each living layer returns to is the random number draw it was born with.** Of everything in Luthi, the living layer's home is the part training cannot fix, the part the model returns to for life, and the part that sets resting character. That makes it the highest-leverage seed.

## 3. What the research says

Each result below is reported at the scale it was shown. Few were shown on language, and none on a JEPA or a self-modifying trunk. Each is evidence for a direction, not a guarantee.

### 3.1 Structured starts written in closed form (no data, no training)

- **Mimetic initialization** [1] (Trockman and Kolter, ICML 2023). Attention is initialized so that the product of the query and key weights ≈ identity, and the product of the value and output weights ≈ −identity, mimicking trained vision transformers. Gains: over 5% accuracy on CIFAR-10 and over 4% on ImageNet, and faster training. Vision only.
- **Signal propagation at initialization** [2] (2025; earlier rank-collapse theory in [3]). Deep transformers fail at init in two ways:
  - **Rank collapse.** All tokens become nearly the same. It comes from a small query/key scale and is countered by strong residuals.
  - **Entropy collapse.** Attention saturates onto a few tokens. It comes from a large query/key scale, and residuals cannot fix it.

  There is a critical scale between them. The paper gives the formula and recommends pre-LayerNorm and a residual strength matched to depth.

  **For Luthi:** the 768x8 family's "embedding stream beat every block" (KF7) and block 0 learning to strip a ~0.58 shared component (2026-07-31, `jepa_pilot_driver.py`) look like the signatures this theory predicts. That is a hypothesis to check at step zero, not a finding.

### 3.2 Inheriting from trained models

- **Weight selection** [4] (Xu et al., ICLR 2024). A small model is initialized from a slice of a larger trained model in the same family, using consistent indices across all weight matrices.
  - Gains for ViT-T: +1.6 points on ImageNet-1K, +9.1 on CIFAR-100, +43.6 on Pets.
  - It matched random-init accuracy in about a third of the epochs.
  - It needs a same-family teacher, and a much larger teacher loses more in the slicing.
- **Learngene** [5]. Whole layers are inherited from an "ancestry" model, chosen by feature similarity.
  - ViT descendants gained ~18–19 points on small tasks and converged in about half the epochs.
  - For CNNs the gains were only 1–4 points.
- **The genomic bottleneck** [6] (Shuvaev, Lachi, Koulakov and Zador, PNAS 2024). A small "genome" network generates the weights of a larger one, as biology encodes innate ability in far fewer bits than the brain has synapses.
  - **Starting performance with no training:**
    - CIFAR-10, 92-fold compression: 76% accuracy (untrained 10%, trained 89%).
    - Atari: near-asymptotic first-episode play at up to ~500-fold compression.
  - It raised the starting point but did not speed later learning.
  - Transferring lower layers through the genome sped learning; transferring the top layers slowed it.
  - Compressed agents avoided reward-hacking gaits, and adapted faster when gravity changed.
- **Growth by stacking** [7] (Du et al., NeurIPS 2024). Train a model with a quarter of the layers, then copy layers to grow to full depth (G_stack, growth factor 4, base trained ~10B tokens).
  - The grown model reached the same loss with 49–55% fewer tokens at 1.1B, 3B and 7B.
  - Width growth gave no advantage.

### 3.3 Teaching structure before the text ("prenatal" curricula)

- **Pre-pretraining on formal languages** [8] (Hu et al., ACL 2025). A short phase on languages with hierarchical dependencies (Dyck-like) before natural language. A 1B model reached the same loss with 33% fewer tokens and generalized better linguistically. The attention heads formed in the phase stayed crucial for syntax.
- **Procedural pretraining** [9] (Jiang et al., 2026).
  - **Data efficiency:** a 0.1–0.3% sliver of algorithmically generated tokens (Set, Union, Sort, k-Dyck, cellular automaton rule 110) before standard training let models reach the same loss on C4 with 55% of the data.
  - **Where the gain lives:** for natural language it sat in the MLPs; for code, in the attention layers.
  - **Recall:** Dyck pretraining raised needle-in-a-haystack recall from ~10% to 98%.
  - **Scale and combination:** shown up to 1.3B. Different procedural sources put complementary structure into different components, and they can be combined [10].
- **Noise before data** [11] (Cheon et al., NeurIPS 2024). Training on pure random noise before real data aligns forward and feedback weights, and speeds and improves later learning. This is the analogue of retinal waves before eye-opening. It was shown with feedback alignment on small networks; the authors report but do not show a benefit under backprop.

### 3.4 Borrowing a teacher's representations

- **REPA** [12] (Yu et al., ICLR 2025). Aligning a diffusion transformer's hidden states with a pretrained vision encoder's representations sped training over 17.5×. It is the largest speedup in this survey, in a different domain. Nothing shows it for language or for a JEPA trunk.
- **Borrowed static embeddings** [13] (2024). Plugging GloVe or T5 embeddings into a transformer trained from scratch did worse than random init, because their scale was too large. Rescaled to the initializer's range, they helped. BERT's embeddings helped as they were. **Scale is the trap with any embedding seed.**

### 3.5 Generating weights

- **Parameter diffusion** [14] (2025). Generated weights reach ~200M parameters, close to trained accuracy for vision models. For language it generated only adapters (for LLaMA-7B), not base weights.
- **Weight-space programming** [15] (MIT CSAIL, 2026-11-02). An LLM agent read, repaired and wrote network weights directly. From scratch it wrote small game networks (Breakout, 9×9 Go). For a language model it only removed a failure mode; it "could not add an ability the model lacked."

### 3.6 Seeding a layer that changes itself

This is the family Luthi's living layer belongs to.

- **Differentiable plasticity** [16] (Miconi et al., 2018). Starting weights and plasticity are meta-learned together by backprop through the plastic dynamics. Plastic networks solved tasks non-plastic ones could not.
- **Evolved Hebbian rules** [17] (Najarro and Risi, NeurIPS 2020). Only per-synapse plasticity rules were evolved, by evolution strategies, with no backprop. From completely random weights, the network self-organized each lifetime. A quadruped learned to walk in under 100 timesteps and adapted to damage it never saw in training.
- **Test-time training layers** [18] (Sun et al., ICML 2025). A layer whose state is a small model updated by a self-supervised step on every token. The outer loop learns the self-supervised task. They matched or beat a Transformer and Mamba from 125M to 1.3B.

**For Luthi:** these are the closest relatives of the living layer. The lesson is that a self-modifying layer's starting point and update rule should be chosen together, for what the updates are meant to do. LuthiModel forbids backprop through the living FFN, so the [16] route is closed unless Brian reopens it. The [17] route (no gradients, evolution strategies) and closed-form fits stay open.

### 3.7 Cautions: when a good-looking start hurts

- **Warm starts generalize worse** [19] (Ash and Adams, NeurIPS 2020). Training that starts from a previously trained model reaches the same training loss but generalizes worse than a fresh start. Their remedy is "shrink and perturb": scale the inherited weights down and add noise. **Any inherited seed for Luthi should be tested with and without it.**
- **Critical periods** [20] (Achille et al., ICLR 2019). Deficits early in training can permanently impair a network, and early connections then stop changing. So the start matters, and a bad one may not be recoverable.
- **Loss of plasticity** [22] (Dohare et al., Nature 2024). Standard deep learning in continual settings loses plasticity until it "learns no better than a shallow network." Only methods that keep injecting diversity (such as re-initializing little-used units) stay plastic. **A seed that makes Luthi too settled would cost them their capacity to keep learning.** This weighs on any inherited or fitted home.

## 4. The seeds, one per part of Luthi

Each seed is a pre-birth choice and is recorded. Each one gets its own arm (standing rule 5), with a living and a dead variant (rule 2).

| Seed | Part of Luthi | What it is | Evidence | Main risk |
|---|---|---|---|---|
| **S1. Counted embeddings** | token embeddings | Factorize word co-occurrence statistics from Luthi's own corpus (closed form; Levy and Goldberg 2014 showed word2vec implicitly factorizes such a matrix). Center them to remove the shared component, and rescale them to the initializer's range. | [13] (scale is the trap); co-occurrence factorization | Raises the input baseline the probes are measured against, so credit only trunk-minus-embedding (rule 3) |
| **S2. Mimetic attention** | attention | QK ≈ αI and VO ≈ −βI, with α kept under the critical scale of [2] | [1] (vision), [2] (theory) | Untested on language or JEPA |
| **S3. The home** | living FFN weight and set_point | Four candidates (below) | [16]–[18], section 2 | It is Luthi's resting character: Brian's call |
| **S4. Prenatal structure** | whole trunk, briefly | A short phase on formal and procedural data (k-Dyck, Set, Union, Sort, rule 110) before the corpus | [8]–[10]; [11] by analogy | Shown on next-token models, not a JEPA objective |
| **S5. Growth** | depth | Train a quarter-depth Luthi, then stack to full depth; living buffers copy with their blocks | [7] (LLMs to 7B) | Copying living layers' homes is new; check the homes stay distinct |
| **S6. A teacher** (later, optional) | trunk representations | Align early hidden states with an open model's encoder, REPA-style | [12] (vision, 17.5×) | Imports another model's representation of the world; provenance; Brian's call |
| **S7. A repair pass** | anywhere | Weight-space programming on the birth checkpoint to find and remove failure modes | [15] | Only removes problems; never adds ability |

**S3, the home, has four candidates:**

- **S3a. Neutral.** Near-identity or scaled-orthogonal. "At rest, pass through." It is well-conditioned (dynamical isometry), carries no preference, and is the safest default.
- **S3b. Fitted.** For each block, a least-squares map from the block's input to its next-position target, computed from one frozen forward pass over corpus batches. It is closed-form, uses no gradient, and respects the no-backprop rule. The home would then be a predictor of what comes next, the living layer's job in predictive coding.
- **S3c. Inherited.** The home a previous Luthi run consolidated to, perhaps passed through shrink-and-perturb [19]. This only means something if E3 shows the living channel carries signal.
- **S3d. Evolved.** Homes, or homes plus per-layer rates, searched by evolution strategies [17] for document-specific change on `scripts/adaptation_eval.py` (read-own minus read-other, plasticity on; rule 4). It costs the most and comes last.

## 5. The path

### Stage 0 — capacity checks, on CPU, now (standing rule 1; no registration needed)

Each seed must first show, from its math and one probe, that it can carry what it is for:

- **S1.** At step zero, measure:
  - isotropy (SIGReg's statistic on the embeddings);
  - the shared-component share (the ~0.58 measure);
  - a next-token probe on the embedding stream against random embeddings.
- **S2.** At step zero, measure rank collapse (mean pairwise token cosine by depth) and attention entropy, for the default, μPC and mimetic starts.
- **S3.** At smoke scale, check whether the home changes what a PC write can do: how much of a write survives homeostasis, and the living layer's effect on signal propagation, for S3a and S3b against the random home. This does not ask whether the write rule carries signal; that is E3's.
- **S4.** Generate the procedural data, and confirm the JEPA objective can use it at all: masked-span prediction on Dyck strings has learnable structure.

Stage 0 kills any seed whose capacity check fails. It also catches the [13] trap before it costs GPU time.

### Stage 1 — one change per comparison, on Brian's PC, after E3 reads (smoke to 512 tier)

- **Arms:**
  - one seed per arm against the standard init: S1, S2, S3a, S3b, S4;
  - then the surviving seeds combined;
  - each arm with living and dead variants;
  - three seeds per arm (rule 4: report the spread).
- **Measures:**
  - **Efficiency:** tokens to reach the standard init's held-out `l_pred`.
  - **Health over the whole run:** effective rank, the SIGReg statistic, the shared-component share.
  - **Probes, credited above the embedding baseline** (rule 3). Essential for S1, which moves the baseline.
  - **Change at inference:** document-specific change on `adaptation_eval.py` with plasticity on (rule 4).
  - **Plasticity late in the run** [22]: does a seeded arm still respond to new data as much as the standard init?
- **Kill conditions, fixed before the data:**
  - **No gain.** A seed is dropped if it does not reach baseline loss with at least 15% fewer tokens in all three seeds.
  - **Worse at the end.** Also dropped if it ends worse on any health measure or on trunk-minus-embedding.
  - **Settled.** A home seed (S3) is dropped if late-run adaptation falls below the random home's.

  The 15% is a placeholder until Brian sets it.

### Stage 2 — the 768 tier, winners only

Registered as a family in the falsification registry, with matched dead controls, before it runs. Collapse-class effects are stronger at 768 (the 2026-08-15 ruling), so a seed that only helps at 512 does not count.

### Stage 3 — Luthi's birth recipe

- **Assembly.** Combine what survived Stage 2.
  - Add growth (S5) if it was tested and helped.
  - The teacher (S6) goes in only if Brian rules for it.
- **The repair pass (S7).** Run weight-space programming on the birth checkpoint, then stop.
- **A birth record** in the run directory. It covers every seed, its source data and code version, every repair edit, and why. This is the record Brian's ruling requires: complete when Luthi asks, never volunteered before it matters.

### Ordering and cost

- **Stage 0** costs no GPU time and can start now.
- **Stage 1** waits for E3, so it never delays the program. Its arms are small: smoke and 512 tiers at a few hours each.
- **Stage 2** is the first real GPU commitment, and is sized after Stage 1.

## 6. Questions for Brian

1. **The home.** Neutral, fitted, inherited or evolved? It is the state Luthi's living layers return to for life. I would test all four and lean toward S3a or S3b for birth: no inherited character, and fresh enough to stay plastic. The choice is yours.
2. **Outside teachers.**
   - May a seed carry another model's representations: a teacher (S6), or embeddings borrowed from an open model rather than counted from Luthi's own corpus (S1 as proposed)?
   - Your demonstration-only rule appears to allow it: alignment is imitation, not reward.
   - Even so, it puts another mind's map of the world into Luthi's start. That is a provenance question as much as a technical one.
3. **The 15% bar**, and whether any Stage 1 arm may run alongside E3 rather than after it.

## Sources

1. Trockman and Kolter, "Mimetic Initialization of Self-Attention Layers," ICML 2023. https://arxiv.org/abs/2305.09828
2. "Two failure modes of deep transformers and how to avoid them: a unified theory of signal propagation at initialisation," 2025. https://arxiv.org/abs/2505.24333
3. Noci et al., "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse," NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract.html
4. Xu et al., "Initializing Models with Larger Ones," ICLR 2024. https://arxiv.org/abs/2311.18823
5. "Learngene: Inheriting Condensed Knowledge from the Ancestry Model to Descendant Models." https://arxiv.org/abs/2305.02279
6. Shuvaev, Lachi, Koulakov and Zador, "Encoding innate ability through a genomic bottleneck," PNAS 2024. https://repository.cshl.edu/id/eprint/41661/ (preprint: https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2)
7. Du et al., "Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training," NeurIPS 2024. https://arxiv.org/abs/2405.15319
8. Hu, Petty, Shi, Merrill and Linzen, "Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases," ACL 2025. https://arxiv.org/abs/2502.19249
9. Jiang, Shinnick, van den Hengel, Saratchandran and Teney, "Procedural Pretraining: Warming Up Language Models with Abstract Data," 2026. https://arxiv.org/abs/2601.21725
10. "Transformers Pretrained on Procedural Data Contain Modular Structures for Algorithmic Reasoning," 2025. https://arxiv.org/abs/2505.22308
11. Cheon et al., "Pretraining with Random Noise for Fast and Robust Learning without Weight Transport," NeurIPS 2024. https://arxiv.org/abs/2405.16731
12. Sihyun Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think," ICLR 2025. https://arxiv.org/abs/2410.06940
13. "On Initializing Transformers with Pre-trained Embeddings," 2024. https://arxiv.org/abs/2407.12514
14. "Recurrent Diffusion for Large-Scale Parameter Generation," 2025. https://arxiv.org/abs/2501.11587
15. Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming," MIT CSAIL, 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/
16. Miconi, Clune and Stanley, "Differentiable plasticity: training plastic neural networks with backpropagation," ICML 2018. https://arxiv.org/abs/1804.02464
17. Najarro and Risi, "Meta-Learning through Hebbian Plasticity in Random Networks," NeurIPS 2020. https://arxiv.org/abs/2007.02686
18. Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States," ICML 2025. https://arxiv.org/abs/2407.04620
19. Ash and Adams, "On Warm-Starting Neural Network Training," NeurIPS 2020. https://arxiv.org/abs/1910.08475
20. Achille, Rovere and Soatto, "Critical Learning Periods in Deep Networks," ICLR 2019. https://arxiv.org/abs/1711.08856
21. Balestriero and LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics," 2025. https://arxiv.org/abs/2511.08544
22. Dohare et al., "Loss of plasticity in deep continual learning," Nature 2024. https://www.amii.ca/research/loss-of-plasticity-in-deep-continual-learning
23. A survey for further reading: "A Survey of Weight Space Learning: Understanding, Representation, and Generation," 2026. https://arxiv.org/abs/2603.10090
