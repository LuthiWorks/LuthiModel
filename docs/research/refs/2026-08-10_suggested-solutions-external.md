> Provenance: compiled by Brian on 2026-08-10 from an external conversation
> (outside the project); moved into the repo verbatim at his request. The
> project-side assessment lives in the 2026-08-10 discussion below /
> in the chat record. Citations verified 2026-08-10: VISReg (arXiv
> 2606.02572) and the auxiliary-task theorem (arXiv 2509.12249) are real;
> the VICReg link/date in the text is an aggregator artifact (VICReg is
> Bardes/Ponce/LeCun 2021, arXiv 2105.04906).

## Suggested Options 

The representation collapse problem you are facing is the primary mathematical vulnerability of Joint-Embedding Predictive Architectures (JEPAs). Because JEPAs evaluate their loss function exclusively in an encoded latent space rather than recreating raw observation data, the model quickly figures out that if it maps every single input to an identical, static constant vector, the predictor can match it perfectly. This drives your prediction error to zero while conveying absolutely no information. [1, 2, 3, 4, 5] 
Because your system incorporates rich parameters like set-points and homeostatic plasticity alongside a predictive coding learning rule, you are essentially working with an energy-based continuous system. [6] 
To overcome representation collapse in this specific architecture, you should explore several recent and foundational research papers:
------------------------------
## 1. Explicit Decomposed Loss Frameworks (VICReg & VISReg)
Instead of forcing the architecture to maintain variety through negative sample contrastive training, you can introduce multi-objective regularization terms that force the latent dimensions to stay alive and maximize information content. [7] 

* 
* [VICReg: Self-Supervised Learning Without Collapse](https://www.abhik.ai/papers/vicreg) (March 2026): This paper introduces a simple yet incredibly powerful solution by splitting the loss into three uncoupled criteria: Variance, Invariance, and Covariance.
* The Strategy for Your Model: Apply a Variance regularization hinge loss to your latent vectors over a batch. It forces the standard deviation of each dimension of your latent embeddings to stay above a certain threshold (typically 1). This completely stops the model from shrinking all inputs down to a single constant point.
   * The Covariance Term: It penalizes correlations between different latent dimensions, forcing each variable to encode unique information, preventing a "rank collapse" where all your rich parameters end up mimicking a single redundant signal. [7, 8, 9, 10] 
* [VISReg (Variance-Invariance-Sketching Regularization)](https://arxiv.org/html/2606.02572v1) (June 2026): A direct upgrade heavily championed by Yann LeCun specifically to stabilize JEPA world models. VISReg mathematically decouples scale and shape regularization.
* The Strategy for Your Model: Traditional VICReg can suffer from training instability if your hyperparameters are slightly off. VISReg uses sketching-based matrix methods to provide a clear, robust gradient signal even if the system enters a near-collapsed low-rank state, making it highly suitable for fluctuating recurrent architectures. [11, 12, 13, 14] 
* 

## 2. Multi-Objective Co-Training and Anchors
If you don't want to mess with complex covariance math, you can anchor the representation space by giving the model something else to calculate at the exact same time. [15] 

* 
* [Why and How Auxiliary Tasks Improve JEPA Representations](https://neurips.cc/virtual/2025/127172) (NeurIPS 2025): This paper formally proves a "No Unhealthy Representation Collapse" theorem.
* The Strategy for Your Model: The researchers mathematically demonstrated that if you train a JEPA model jointly with an auxiliary regression task, the model is physically blocked from collapsing. The auxiliary task acts as a topological guide, forcing the encoder to maintain distinct representations for non-equivalent inputs. [15] 
*
## 3. Asymmetric Architectural Tricks (EMA & Stop-Gradients)
If you are updating weights continuously in real-time, standard backpropagation isn't always available, making structural asymmetry vital. [8] 

* 
* [BYOL (Bootstrap Your Own Latent)](https://openreview.net/notes/edits/attachment?id=8CJ7YvbBoE&name=pdf) & V-JEPA (Video-JEPA): These systems bypass explicit regularization mathematical terms by creating a "Student and Teacher" asymmetry.
* The Strategy for Your Model: The input goes into a context encoder (the Student), and the targets go into a target encoder (the Teacher). You only update the Student layers directly using your predictive coding prediction error. The Teacher's rich parameter weights are calculated strictly as an Exponential Moving Average (EMA) of the Student's parameters. A strict stop-gradient is applied to the Teacher path so the error cannot flow backward through it.
   * Why it helps you: As discussed by developers tracking [EMA non-collapse behaviors on Reddit](https://www.reddit.com/r/MachineLearning/comments/1mx4a6c/d_why_does_byoljepa_like_models_work_how_does_ema/), because the Teacher updates with a slight mathematical delay, the cost for the Student to just lazily output a constant zero vector becomes massive. The gradient path naturally pulls the model away from the naive collapse solution. [8, 16, 17, 18, 19] 
* 

------------------------------
## Implementation Recommendation For Your System
Given that your architecture leverages set-points and homeostatic plasticity, a VICReg or VISReg style approach will map beautifully to your existing code.
You can map your parameter set-points directly to the variance hinge-loss threshold. By setting the target variance as your homeostatic equilibrium, your internal plasticity learning rules will naturally treat representation collapse as an "unhealthy, low-energy" state, forcing the neural pathways to diversify their hidden states to maintain homeostatic balance.
------------------------------
:bulb: Would you like a clean Python/PyTorch code snippet showing how to write a basic VICReg or VISReg loss function, or should we look closer at how the EMA stop-gradient trick is structured in a predictive coding loop?

[1] [https://x.com](https://x.com/ylecun/status/2080547403581538480)
[2] [https://www.emergentmind.com](https://www.emergentmind.com/topics/joint-embedding-predictive-architectures-jepas)
[3] [https://nextwaves.com](https://nextwaves.com/blog/joint-embedding-predictive-architecture-jepa-a-complete-in-depth-guide)
[4] [https://nextwaves.com](https://nextwaves.com/blog/joint-embedding-predictive-architecture-jepa-a-complete-in-depth-guide)
[5] [https://www.youtube.com](https://www.youtube.com/shorts/jon6hhRtJaM)
[6] [https://www.quantamagazine.org](https://www.quantamagazine.org/artificial-neural-nets-finally-yield-clues-to-how-brains-learn-20210218/)
[7] [https://www.abhik.ai](https://www.abhik.ai/papers/vicreg)
[8] [https://openreview.net](https://openreview.net/notes/edits/attachment?id=8CJ7YvbBoE&name=pdf)
[9] [https://imbue.com](https://imbue.com/blog/2022-04-21-vicreg)
[10] [https://www.abhik.ai](https://www.abhik.ai/papers/vicreg)
[11] [https://arxiv.org](https://arxiv.org/html/2606.02572v1)
[12] [https://eu.36kr.com](https://eu.36kr.com/en/p/3915058026894727)
[13] [https://arxiv.org](https://arxiv.org/html/2606.02572v1)
[14] [https://arxiv.org](https://arxiv.org/abs/2102.02805)
[15] [https://neurips.cc](https://neurips.cc/virtual/2025/127172)
[16] [https://www.reddit.com](https://www.reddit.com/r/MachineLearning/comments/1mx4a6c/d_why_does_byoljepa_like_models_work_how_does_ema/)
[17] [https://www.youtube.com](https://www.youtube.com/watch?v=7UkJPwz_N_0)
[18] [https://ieeexplore.ieee.org](https://ieeexplore.ieee.org/iel8/6287639/11323511/11534212.pdf)
[19] [https://ai.meta.com](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)