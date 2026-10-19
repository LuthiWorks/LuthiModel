"""Output-level episodic memory for the hybrid block.

The episode store is the "hippocampus" of the architecture. It records
(context, output) pairs and retrieves matching outputs to blend with
current computation. This is separate from the layer-level episodes
inside LivingLayerV6.

The episode store carries most of the recall weight — in-weight memory
is weak by comparison (proven in the research phase). The two mechanisms
are complementary: living weights provide temporal dynamics, the episode
store provides accurate recall.

WHAT THIS STORE ACTUALLY DOES IN A BATCHED TRUNK (measured 2026-10-19,
docs/research/2026-10-19_the-living-write-is-mean-field.md). The context
is the input pooled over BATCH and SEQUENCE, and the stored value is the
output pooled the same way. With shuffled training batches those pools are
near-identical from batch to batch, salience (mean |output|) clears the
0.1 threshold on every call, and the best match clears the recall gate on
every call. So in every living v2 block this store fills within its first
`num_episodes` forwards and then, on nearly every forward (measured duty
1.000 on a trained smoke checkpoint) -- training and eval, frozen or not --
replaces the block output with
``(1 - blend) * output + blend * (a stored batch-mean output)``: it shrinks
every token's deviation from the mean by ``1 - blend`` (0.7 at the default
0.3) and re-injects a batch-constant vector, once per block. The dead-FFN
control arm has no such store, so this is an uncontrolled difference in
every living-vs-dead comparison the registry has made. The counters below
make it observable; ``PredictiveCodingBlock(block_episode_store="off")``
removes it for a clean comparison.
"""

import torch
import torch.nn as nn


class EpisodeStore(nn.Module):
    """Output-level episodic memory with context-gated retrieval.

    Stores experiences as (context_signature, output_vector, salience) tuples.
    When a new input's context matches a stored experience, the stored output
    is blended with the current output.

    Survives 100+ interfering experiences (validated in research phase).
    """

    def __init__(
        self,
        d_model: int,
        num_episodes: int = 64,
        context_dim: int = 64,
        blend_factor: float = 0.3,
        salience_threshold: float = 0.1,
        recall_similarity_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.blend_factor = blend_factor
        self.salience_threshold = salience_threshold
        self.recall_similarity_threshold = recall_similarity_threshold

        # Item #6 (2026-06-28): set by freeze_plasticity() during the lived
        # JEPA re-encode. When True, forward() still recalls + blends (that
        # contribution is part of the representation the lived gradient
        # must reproduce) but SKIPS store(), so the learner's re-encode does
        # not WRITE a second episode for a context perception already
        # stored. NB this closes the double-WRITE only: recall() still READS
        # these buffers, and under the async learner (Plan §4) the actor's
        # perception-time store() writes them concurrently -- that read-race
        # is closed by §4's snapshot-under-lock, not by this flag.
        self._plasticity_frozen: bool = False

        # Random projection for context compression (fixed)
        proj = torch.randn(d_model, context_dim) / (d_model**0.5)
        self.register_buffer("context_proj", proj)

        # Episode storage
        self.register_buffer(
            "episode_contexts", torch.zeros(num_episodes, context_dim)
        )
        self.register_buffer(
            "episode_outputs", torch.zeros(num_episodes, d_model)
        )
        self.register_buffer("episode_saliences", torch.zeros(num_episodes))
        self.register_buffer("episode_count", torch.tensor(0, dtype=torch.long))

        # Observational counters (2026-10-19). Plain Python, deliberately not
        # buffers: the v2 loader is strict=True and a new persistent buffer
        # would break every existing checkpoint. Per-process, so they reset
        # on resume -- they are differenced between records to get rates.
        self.forward_calls: int = 0
        self.recall_fires: int = 0
        self.store_writes: int = 0
        self.last_recall_sim: float = float("nan")
        # Share of the blended output's mean token norm contributed by the
        # recalled vector: blend * ||recalled|| / mean_t ||output_t||.
        self.last_blend_share: float = 0.0
        self._last_best_sim: float = float("nan")

    def compute_context(self, x: torch.Tensor) -> torch.Tensor:
        """Compute context signature from input via random projection.

        Args:
            x: [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            [context_dim] L2-normalized context vector.
        """
        # Pool across batch and sequence dimensions
        if x.dim() == 3:
            pooled = x.mean(dim=(0, 1))  # [d_model]
        else:
            pooled = x.mean(dim=0)  # [d_model]

        ctx = pooled @ self.context_proj  # [context_dim]
        return ctx / (ctx.norm() + 1e-8)

    def recall(self, context: torch.Tensor) -> torch.Tensor | None:
        """Retrieve best-matching stored output.

        Args:
            context: [context_dim] L2-normalized context vector.

        Returns:
            [d_model] recalled output, or None if no good match.
        """
        if self.episode_count == 0:
            return None

        n = self.episode_count.item()
        stored_contexts = self.episode_contexts[:n]  # [n, context_dim]

        # Cosine similarity (both are L2-normalized)
        sims = torch.mm(stored_contexts, context.unsqueeze(-1)).squeeze(-1)  # [n]
        best_idx = sims.argmax()
        best_sim = sims[best_idx]
        self._last_best_sim = float(best_sim.item())

        if best_sim < self.recall_similarity_threshold:
            return None

        # Return recalled output scaled by similarity
        return self.episode_outputs[best_idx] * best_sim

    def store(
        self, context: torch.Tensor, output: torch.Tensor, salience: float
    ) -> None:
        """Store an experience as a new episode.

        Uses salience-based pruning when at capacity.

        Args:
            context: [context_dim] L2-normalized context vector.
            output: [d_model] output vector to store (mean-pooled from sequence).
            salience: Scalar salience score for this experience.
        """
        if salience < self.salience_threshold:
            return

        n = self.episode_count.item()

        if n < self.num_episodes:
            idx = n
            self.episode_count.add_(1)
        else:
            min_idx = self.episode_saliences[:n].argmin()
            if salience <= self.episode_saliences[min_idx]:
                return
            idx = min_idx.item()

        self.episode_contexts[idx] = context.detach()
        self.episode_outputs[idx] = output.detach()
        self.episode_saliences[idx] = salience
        self.store_writes += 1

    def forward(
        self, x_input: torch.Tensor, x_output: torch.Tensor
    ) -> torch.Tensor:
        """Recall matching episode and blend with current output, then store.

        Args:
            x_input: [batch, seq_len, d_model] block input (for context).
            x_output: [batch, seq_len, d_model] current block output (to blend).

        Returns:
            [batch, seq_len, d_model] blended output.
        """
        with torch.no_grad():
            context = self.compute_context(x_input)
            salience = x_output.detach().abs().mean().item()

            # Recall
            self.forward_calls += 1
            recalled = self.recall(context)
            self.last_recall_sim = self._last_best_sim

            # Store the current output (mean-pooled) as a new episode.
            # Skipped under frozen plasticity (Item #6): the lived
            # re-encode must not write a second episode for a context
            # perception already stored, and under the async learner (§4)
            # must not race the actor's perception-time buffer writes.
            if not self._plasticity_frozen:
                output_pooled = x_output.detach().mean(dim=(0, 1))  # [d_model]
                self.store(context, output_pooled, salience)

        # Blend recalled output with current output
        if recalled is not None:
            # Broadcast recalled [d_model] to match output shape
            blend = recalled.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
            x_output = (1 - self.blend_factor) * x_output + self.blend_factor * blend
            self.recall_fires += 1
            with torch.no_grad():
                tok_norm = x_output.detach().norm(dim=-1).mean()
                self.last_blend_share = float(
                    (self.blend_factor * recalled.detach().norm()
                     / tok_norm.clamp(min=1e-12)).item()
                )
        else:
            self.last_blend_share = 0.0

        return x_output

    def stats(self) -> dict[str, float]:
        """Return diagnostic metrics."""
        return {
            "episodes_stored": self.episode_count.item(),
            "mean_salience": (
                self.episode_saliences[: self.episode_count].mean().item()
                if self.episode_count > 0
                else 0.0
            ),
            # 2026-10-19: recall duty and blend share. recall_fires ==
            # forward_calls with blend_share > 0 is the "blends a constant
            # into every forward" signature.
            "forward_calls": int(self.forward_calls),
            "recall_fires": int(self.recall_fires),
            "store_writes": int(self.store_writes),
            "recall_duty": (
                self.recall_fires / self.forward_calls
                if self.forward_calls else float("nan")
            ),
            "last_recall_sim": float(self.last_recall_sim),
            "last_blend_share": float(self.last_blend_share),
        }
