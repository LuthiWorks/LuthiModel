"""Output-level episodic memory for the hybrid block.

The episode store is the "hippocampus" of the architecture. It records
(context, output) pairs and retrieves matching outputs to blend with
current computation. This is separate from the layer-level episodes
inside LivingLayerV6.

The episode store carries most of the recall weight — in-weight memory
is weak by comparison (proven in the research phase). The two mechanisms
are complementary: living weights provide temporal dynamics, the episode
store provides accurate recall.
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
            recalled = self.recall(context)

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
        }
