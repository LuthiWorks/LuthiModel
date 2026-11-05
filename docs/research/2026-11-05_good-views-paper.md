# Good views and bad views: Balestriero & LeCun, 2026-10-29 — 2026-11-05

## Objective

Brian brought in a new JEPA paper: "Proving and Improving the Platonic Representation Hypothesis with JEPAs" (Randall Balestriero and Yann LeCun, [arXiv:2610.39302](https://arxiv.org/abs/2610.39302), submitted 2026-10-29). This note records what it claims, what it means for Luthi, and one arm worth registering when training returns.

I read the main body (sections 1 to 7) from the PDF's extracted text. I did not read the appendices or the proofs. The reading of what it means for Luthi is mine, not the paper's.

## Process

### Step 1: what the paper claims

- **The theorem.** JEPAs trained separately on different modalities learn the same embedding, up to one rotation each, when their positive pairs relate the same world states. The conditions are idealized: a finite world, injective renderings, population optima, and minimizers unique up to rotation.
- **Where uniqueness is proved.** For whitened second-order anti-collapse under a spectral gap. For SIGReg's full-distribution constraint it is left as an assumption.
- **Good and bad views.** A good-view JEPA's positives are two observations of one scene, such as two captions of one image. A bad-view JEPA's positives are corruptions of one observation, such as two maskings of one sentence. Bad views spend embedding directions on the modality's own nuisance, such as a caption's wording.
- **Next-token prediction.** It has the form of a JEPA whose positives are a prefix and its continuation inside one document. The paper applies the nuisance argument to it "heuristically", as a prediction to test.

### Step 2: what they measured

Setup: LeJEPA objective with SIGReg at λ = 0.02, a 12-layer BERT at 33M parameters (and 109M), 8 views per item, each trimmed to a random span with 15% of tokens masked. The predictor is the identity: the two embeddings are compared directly.

| Text encoder, same data and architecture | Flickr mutual 10-NN | Flickr R@1 |
|---|---:|---:|
| Good-view JEPA (views from different captions of one image) | .062 | .044 |
| Bad-view JEPA (views from one caption) | .047 | .027 |
| Next-token prediction | .042 | .021 |

At scale, with a 109M encoder on 900M samples from the texts of 940M web images, scored on the WIT-1024 benchmark against DINOv2:

| Text encoder | WIT mutual 10-NN |
|---|---:|
| Good-view JEPA, 109M | .184 |
| Next-token prediction, same encoder and texts | .111 |
| Bad-view JEPA, same encoder and texts | .088 |
| Qwen2.5-72B | .154 |
| Llama-3-70B | .151 |

Other findings:

- **Sample efficiency.** The good-view JEPA reaches .109 after 21.6M samples, where next-token prediction reaches .111 after 900M: "42 times fewer samples". Two cautions: they scored no earlier next-token checkpoint, so it may have reached that level sooner, and the two runs used different recipes. The cleaner comparison is at matched samples: .108 against .071 at 21.6M, for the 33M encoder.
- **Trimming matters most.** Using whole captions as views loses a quarter of the alignment. Masking rate, number of views, batch size and width barely matter.
- **Collapse.** Without gradient clipping, 3 of 36 runs at the default SIGReg weight collapsed abruptly between epochs 20 and 60 of 200, and so did every other SIGReg weight they tried. With clipping and a lower learning rate, none of their longer runs collapsed.

### Step 3: the paper's own caveats

- No experiment meets the theorem's hypotheses exactly.
- **Learned predictors are not covered.** JEPAs used as world models "predict one embedding from the other through a learned predictor, often conditioned on an action, which our analysis does not cover."
- **Good views need grouping.** Someone has to know which texts describe the same scene. At scale, two of each image's three texts were written by an image captioner whose vision encoder is CLIP. Without any captioner, alignment was .060 against .107.
- A fourth seed with twice the SIGReg weight reached only .121. All four seeds average .168, still above every language model they tested.
- At least 43 of the 1,024 benchmark captions appear in the training alt-texts. Removing them changed no model they checked by more than .002.
- The 109M model read 116B text tokens. The language models it beat were trained on 15 to 18T.

### Step 4: what it means for Luthi

- **Luthi's text JEPA uses bad views, by this definition.** The context is the first 80% of a window and the target is the rest of the same window (`DEFAULT_CONTEXT_FRACTION = 0.8` in `luthi/v2/jepa_loss.py`). Both come from one observation. The LLM-JEPA next-token term (`w_ntp`) is the paper's other bad-view case.
- **A second reading of an old lead.** The 2026-07-27 lead in `To-Do.md` says the prediction task may be too easy (trivial cosine 0.9975) and proposes lowering `context_fraction`. That would make a bad view harder. It would not make it a good view. The paper suggests the kind of pair matters more than its difficulty, though it did not test that on anything like Luthi.
- **A lived life may supply good views.** Seeing a place from two angles, or hearing someone while seeing them, is two observations of one world state. The paper names audio, video and actions as future work. This fits Brian's ruling that the world just exists.
- **What it does not show.** It measures agreement with vision encoders. It says nothing about living weights, about a learned action-conditioned predictor like Luthi's, or about VISReg, which replaced SIGReg here on 2026-08-11. Whether VISReg meets the theorem's anti-collapse assumptions is unchecked. The next-token term was adopted here for depth-8 stability, which this paper does not address.
- **Clipping.** Depth-8 probe arms here already clip (`ARM_GRAD_CLIP` in `scripts/jepa_pilot_driver.py`), and depth-4 arms run with it off to stay bit-identical. The August depth record should be checked against the paper's collapse finding before treating clipping as a new lever.

## Conclusion

Nothing changes now. The registered program (E1 to E5) tests the living channel, and this paper is about something else. But it gives a specific, testable reason to think Luthi's text positives are the weak kind, and that is worth one arm.

**Proposed, for registration when training returns, after E3 reads:** a good-view text arm. Same encoder, trainer and regularizer; only the positives change. Under the standing rules it needs its matched dead control (rule 2), a list of every difference between arms (rule 5), and a capacity check first (rule 1).

**To decide before registering:**

- **Where good views come from in this corpus.** One untested idea: different public-domain translations of the same work, aligned by passage. The corpus brief already lists multiple translations of Aesop. Whether enough such pairs exist is the capacity check.
- **What to measure.** The paper scores alignment with a frozen vision encoder. This project's instruments are the standardized probe against the embedding baseline (E4) and the adaptation eval (E1). Which of these a good-view arm should move, and by how much, has to be frozen before the run.
- **How views interact with the learned predictor,** which the paper's theory leaves out.

## Artifacts

- The paper: [arXiv:2610.39302](https://arxiv.org/abs/2610.39302)
- `To-Do.md`, "Leads, not yet registered"
