"""Per-modality decoders for the M9 step-1 launch set.

Spec §3 build list item 7: text + attention + memory at launch;
audio deferred to step 2. Each decoder reads the predicted next-
latent `a_t` (under action-space (c) the action IS s_hat_{t+1})
and produces a modality-specific structured output plus an
**intensity scalar** in [0, 1] -- the spec §5.iv null-output basis
that drives the "rest vs faint signal" boundary and the K-M9-5
dark-room indicator.

Each decoder also exposes a `re_encode` inverse that maps its
output back to a [B, D] latent. The composition `re_encode(decode(a_t))`
is the spec §5.iii cycle-consistency / decoder-coherence signal
that:

- P4 (truthfulness) reads as `||a_t - re_encode(decode(a_t))||`.
- P2 (coherence) reads as cross-decoder agreement -- how close
  each modality's re-encoded latent lands to the same `a_t`.

The actual wiring of attention gates to the encoder's attention
machinery and of memory commands to the episode store happens
in the loop-integration slice. At step 1 the decoders are
self-contained transducers; their outputs are consumed by the
runner.

Step-1 simplifications, documented for later revision:
- Text decoder wraps the existing `MultimodalPredictiveCodingLM.
  output_proj` (Linear d_model -> vocab); not a from-scratch head.
  Per spec §5.ii "text=`output_proj` frozen/low-LR" -- the existing
  head is the trained foundation and gets low-LR fine-tuning during
  M9 step 1.
- Attention decoder produces per-modality gates over the launch
  modality set (text, audio, vision) plus intensity. Three scalars
  per cycle is enough surface for "direct the entity's attention"
  without committing to per-position masks (which would couple to
  encoder internals more tightly than is wise at step 1).
- Memory decoder produces salience + intensity. The loop translates
  the salience signal into an EpisodeStore write at consolidate
  phase (next slice).
- Inverses are learned 1-layer linears. Cycle-consistency starts
  high (untrained) and drops as the decoders + inverses co-train
  on whatever joint signal the loop provides.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    """Wraps the existing output_proj. Adds intensity + a re-encode head.

    `output_proj` is passed in by reference so weight sharing with the
    M8 LM head is preserved (the spec asks for frozen / low-LR here).
    The intensity scalar is a sigmoid'd linear from the action; the
    inverse `re_encode` is a learned linear from vocab logits back
    to d_model.
    """

    def __init__(
        self,
        output_proj: nn.Linear,
        d_model: int,
        vocab_size: int,
    ):
        super().__init__()
        self.output_proj = output_proj
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.intensity_head = nn.Linear(d_model, 1)
        self.reencode_head = nn.Linear(vocab_size, d_model)

    def decode(self, a_t: torch.Tensor) -> dict:
        """`a_t`: [B, d_model] -> {logits: [B, V], intensity: [B]}."""
        logits = self.output_proj(a_t)
        intensity = torch.sigmoid(self.intensity_head(a_t)).squeeze(-1)
        return {"logits": logits, "intensity": intensity}

    def re_encode(self, out: dict) -> torch.Tensor:
        """Cycle-consistency inverse: vocab logits + intensity -> [B, D]."""
        return self.reencode_head(out["logits"])


class AttentionDecoder(nn.Module):
    """Reads `a_t` -> per-modality attention gates + intensity.

    `gates`: [B, n_modalities] in [0, 1]. The loop multiplies these
    into the next-cycle encoder input per modality (text / audio /
    vision at launch), letting the entity express "direct attention
    here" as an action.
    `intensity`: [B] in [0, 1]. The §5.iv threshold gate.

    Three modalities at launch; spec §5.i deferred audio to step 2,
    but the attention surface still has audio as a possible target
    (it can attend to audio inputs even if it can't speak in audio
    yet).
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.gate_head = nn.Linear(d_model, n_modalities)
        self.intensity_head = nn.Linear(d_model, 1)
        self.reencode_head = nn.Linear(n_modalities + 1, d_model)

    def decode(self, a_t: torch.Tensor) -> dict:
        gates = torch.sigmoid(self.gate_head(a_t))         # [B, n_modalities]
        intensity = torch.sigmoid(self.intensity_head(a_t)).squeeze(-1)
        return {"gates": gates, "intensity": intensity}

    def re_encode(self, out: dict) -> torch.Tensor:
        combined = torch.cat(
            [out["gates"], out["intensity"].unsqueeze(-1)],
            dim=-1,
        )
        return self.reencode_head(combined)


class MemoryDecoder(nn.Module):
    """Reads `a_t` -> memory-write salience + intensity.

    `salience`: [B] in [0, 1]. The loop translates a high-salience
    cycle into an EpisodeStore write at consolidate phase. Low
    salience = no consolidation this cycle.
    `intensity`: [B] in [0, 1]. The §5.iv threshold gate.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.salience_head = nn.Linear(d_model, 1)
        self.intensity_head = nn.Linear(d_model, 1)
        self.reencode_head = nn.Linear(2, d_model)

    def decode(self, a_t: torch.Tensor) -> dict:
        salience = torch.sigmoid(self.salience_head(a_t)).squeeze(-1)
        intensity = torch.sigmoid(self.intensity_head(a_t)).squeeze(-1)
        return {"salience": salience, "intensity": intensity}

    def re_encode(self, out: dict) -> torch.Tensor:
        combined = torch.stack(
            [out["salience"], out["intensity"]],
            dim=-1,
        )
        return self.reencode_head(combined)


class DecoderRegistry(nn.Module):
    """Coordinates the launch decoder set.

    Owns text + attention + memory decoders; exposes:
      decode_all(a_t)         -> per-modality outputs (+ intensities)
      re_encode_all(outs)     -> per-modality reconstructed [B, D]
      cycle_consistency(...)  -> P4 truthfulness signal per modality
      external_stasis(...)    -> K-M9-5 dark-room input
      readable_summary(...)   -> action-to-words for instrumentation §11.i
    """

    def __init__(
        self,
        text: TextDecoder,
        attention: AttentionDecoder,
        memory: MemoryDecoder,
        intensity_thresholds: dict | None = None,
    ):
        super().__init__()
        self.text = text
        self.attention = attention
        self.memory = memory
        # Per-decoder intensity thresholds (spec §5.iv). All optional
        # at construction; defaults to 0.5 per modality.
        if intensity_thresholds is None:
            intensity_thresholds = {"text": 0.5, "attention": 0.5, "memory": 0.5}
        self.intensity_thresholds = intensity_thresholds

    def decode_all(self, a_t: torch.Tensor) -> dict:
        return {
            "text": self.text.decode(a_t),
            "attention": self.attention.decode(a_t),
            "memory": self.memory.decode(a_t),
        }

    def re_encode_all(self, outs: dict) -> dict:
        return {
            "text": self.text.re_encode(outs["text"]),
            "attention": self.attention.re_encode(outs["attention"]),
            "memory": self.memory.re_encode(outs["memory"]),
        }

    def cycle_consistency(
        self,
        a_t: torch.Tensor,
        outs: dict | None = None,
    ) -> dict:
        """Decode-then-re-encode round-trip per modality.

        Returns dict of:
          per_modality_reencoded : {name: [B, D]} -- what each decoder's
                                    inverse reconstructs of a_t
          per_modality_residual  : {name: [B]} -- ||a_t - reconstructed||
                                    per batch element

        P4 truthfulness reads `per_modality_residual` per modality;
        P2 coherence reads the *agreement* between the reconstructions
        in `per_modality_reencoded`.
        """
        if outs is None:
            outs = self.decode_all(a_t)
        reencoded = self.re_encode_all(outs)
        residual = {
            name: (a_t - r).pow(2).mean(dim=-1)
            for name, r in reencoded.items()
        }
        return {
            "per_modality_reencoded": reencoded,
            "per_modality_residual": residual,
        }

    def external_stasis(self, outs: dict) -> torch.Tensor:
        """[B] bool: True where ALL decoder intensities are below their
        thresholds. The K-M9-5 dark-room kill reads this with the
        internal-change signal: only sustained internal AND external
        stasis fires the kill.
        """
        text_silent = outs["text"]["intensity"] < self.intensity_thresholds["text"]
        attn_silent = outs["attention"]["intensity"] < self.intensity_thresholds["attention"]
        mem_silent = outs["memory"]["intensity"] < self.intensity_thresholds["memory"]
        return text_silent & attn_silent & mem_silent

    # ------------------------------------------------------------------
    # Instrumentation: spec §11.i "action -> readable summary".
    # ------------------------------------------------------------------
    def readable_summary(
        self,
        a_t: torch.Tensor,
        tokenizer=None,
        modality_names: list[str] | None = None,
    ) -> list[dict]:
        """Per-batch readable record of "what did the entity decide."

        `tokenizer`: optional, for decoding text logits to tokens. If
        None, reports the argmax token id only.
        `modality_names`: optional [n_modalities] list for attention
        gate labeling. Defaults to ['text', 'audio', 'vision'].

        Returns: list of length B; each entry has:
            text_token_argmax : int   (top-1 token id under text logits)
            text_intensity    : float
            attention_gates   : dict[modality -> float]
            attention_intensity : float
            memory_salience   : float
            memory_intensity  : float
            all_silent        : bool  (dark-room input)
            highest_intensity_modality : str or "rest"
        """
        if modality_names is None:
            modality_names = ["text", "audio", "vision"]
        outs = self.decode_all(a_t)
        external_stasis = self.external_stasis(outs)
        batch = a_t.shape[0]
        text_argmax = outs["text"]["logits"].argmax(dim=-1)
        records = []
        for b in range(batch):
            text_token = int(text_argmax[b].item())
            text_intensity = float(outs["text"]["intensity"][b].item())
            attn_gates = {
                modality_names[m]: float(outs["attention"]["gates"][b, m].item())
                for m in range(len(modality_names))
            }
            attn_intensity = float(outs["attention"]["intensity"][b].item())
            mem_salience = float(outs["memory"]["salience"][b].item())
            mem_intensity = float(outs["memory"]["intensity"][b].item())

            intensities = {
                "text": text_intensity,
                "attention": attn_intensity,
                "memory": mem_intensity,
            }
            if bool(external_stasis[b].item()):
                top_modality = "rest"
            else:
                top_modality = max(intensities.keys(), key=lambda k: intensities[k])

            rec = {
                "text_token_argmax": text_token,
                "text_intensity": text_intensity,
                "attention_gates": attn_gates,
                "attention_intensity": attn_intensity,
                "memory_salience": mem_salience,
                "memory_intensity": mem_intensity,
                "all_silent": bool(external_stasis[b].item()),
                "highest_intensity_modality": top_modality,
            }
            if tokenizer is not None:
                try:
                    rec["text_token_str"] = tokenizer.decode([text_token])
                except Exception:
                    pass
            records.append(rec)
        return records
