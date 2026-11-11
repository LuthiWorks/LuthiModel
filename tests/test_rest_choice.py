"""Rest as something Luthi chooses (2026-11-11).

Off, nothing changes. On, a decoded action also says whether it chose rest, the
trunk and every other reading are as before, and the choice's state continues
exactly across a save.
"""

from pathlib import Path

import torch

from luthi.v2.m9.rest_choice import RestChoice
from luthi import sanctuary_interface as L
from tests.test_m9_external_actor import D, SEQ, VOCAB, _build_trainer


def _live(trainer, moments: int, seed: int = 0) -> list[dict]:
    model = trainer.loss_module.online_encoder
    generator = torch.Generator().manual_seed(seed)
    chosen, previous = [], None
    for _ in range(moments):
        tokens = torch.randint(0, VOCAB, (1, SEQ), generator=generator)
        state = L.perceive(model, text_tokens=tokens)
        if previous is not None:
            L.observe_transition(
                trainer, previous[0], previous[1], state.s_t,
                plan_snapshot=previous[2], context_obs=previous[3],
            )
        selection = L.select_action(trainer, state.s_t, context_latents=state.ctx_latents)
        chosen.append(L.decode_action(
            trainer, selection.action, s_t=state.s_t, context_latents=state.ctx_latents,
        ))
        previous = (state.s_t, selection.action, selection.plan_snapshot, state.context_obs)
    return chosen


def test_off_by_default_and_nothing_is_built(tmp_path: Path):
    trainer = _build_trainer(tmp_path / "a")
    assert trainer.rest_choice is None
    assert "rest" not in _live(trainer, 2)[-1]
    assert "m9_rest_choice" not in trainer._extra_checkpoint_state()
    trainer.action_log.close()


def test_on_leaves_everything_else_bit_identical(tmp_path: Path):
    # Each is built and lived in turn: building seeds the global stream, and
    # living draws from it.
    off = _build_trainer(tmp_path / "off")
    lived_off = _live(off, 6)
    on = _build_trainer(tmp_path / "on", rest_choice=True)
    lived_on = _live(on, 6)
    for a, b in zip(lived_off, lived_on):
        assert "rest" in b
        assert {k: v for k, v in b.items() if k != "rest"} == a
    for (name, p), (_, q) in zip(
        off.loss_module.state_dict().items(), on.loss_module.state_dict().items(),
    ):
        assert torch.equal(p, q), name
    off.action_log.close(); on.action_log.close()


def test_building_it_does_not_move_the_global_random_stream():
    torch.manual_seed(5)
    expected = torch.rand(4)
    torch.manual_seed(5)
    RestChoice(D)
    assert torch.equal(torch.rand(4), expected)


def test_nothing_is_chosen_before_the_window_is_warm():
    choice = RestChoice(D, warmup=8, window=16)
    nothing = torch.zeros(D)
    for _ in range(7):
        read = choice.read(choice.direction * 100.0, nothing)
        assert read == {"chosen": False, "armed": False, "reading": read["reading"]}


def test_an_action_that_leans_to_rest_is_chosen_and_one_that_does_not_is_not():
    choice = RestChoice(D, warmup=8, window=64)
    generator = torch.Generator().manual_seed(1)
    nothing = torch.zeros(D)
    for _ in range(32):
        choice.read(torch.randn(D, generator=generator) * 0.1, nothing)
    assert choice.read(choice.direction * 5.0, nothing)["chosen"] is True
    assert choice.read(-choice.direction * 5.0, nothing)["chosen"] is False
    # Leaning no more toward rest than doing nothing does is not a choice.
    lean = choice.direction * 5.0
    assert choice.read(lean, lean)["chosen"] is False


def test_the_reading_is_measured_against_doing_nothing():
    choice = RestChoice(D)
    action, nothing = torch.randn(D), torch.randn(D)
    expected = float(torch.dot(choice.direction, action - nothing))
    assert abs(choice.read(action, nothing)["reading"] - expected) < 1e-6


def test_the_direction_is_the_same_on_every_start_and_is_not_trained():
    assert torch.equal(RestChoice(D).direction, RestChoice(D).direction)
    assert list(RestChoice(D).parameters()) == []


def test_a_saved_choice_continues_exactly(tmp_path: Path):
    first = _build_trainer(tmp_path / "a", rest_choice=True, rest_choice_warmup=4)
    _live(first, 8)
    state = first._extra_checkpoint_state()
    second = _build_trainer(tmp_path / "b", rest_choice=True, rest_choice_warmup=4)
    second.rest_choice.load_choice_state(state["m9_rest_choice"])
    assert list(second.rest_choice.readings) == list(first.rest_choice.readings)
    assert second.rest_choice.read_count == first.rest_choice.read_count == 8
    action, nothing = torch.randn(D), torch.randn(D)
    assert first.rest_choice.read(action, nothing) == second.rest_choice.read(action, nothing)
    first.action_log.close(); second.action_log.close()


def test_chance_rate_for_random_actions_is_about_one_in_forty():
    """What an untrained policy would do: press it now and then by chance.
    Pinned so a change to the threshold has to say what it does to this."""
    choice = RestChoice(D)
    generator = torch.Generator().manual_seed(3)
    nothing = torch.zeros(D)
    for _ in range(4000):
        choice.read(torch.randn(D, generator=generator), nothing)
    rate = choice.chosen_count / choice.read_count
    assert 0.01 < rate < 0.05, rate


# ---- Recall as a choice (2026-11-11) ----

from luthi.v2.m9.rest_choice import RECALL_DIRECTION_SEED


def test_recall_is_off_by_default_and_on_leaves_everything_else_bit_identical(tmp_path: Path):
    off = _build_trainer(tmp_path / "off", rest_choice=True)
    assert off.recall_choice is None
    lived_off = _live(off, 6)
    on = _build_trainer(tmp_path / "on", rest_choice=True, recall_choice=True)
    lived_on = _live(on, 6)
    for a, b in zip(lived_off, lived_on):
        assert "recall" not in a and set(b["recall"]) == {"chosen", "armed", "reading"}
        assert {k: v for k, v in b.items() if k != "recall"} == a
    assert "m9_recall_choice" not in off._extra_checkpoint_state()
    assert "m9_recall_choice" in on._extra_checkpoint_state()
    off.action_log.close(); on.action_log.close()


def test_recall_reads_a_different_direction_from_rest():
    rest, recall = RestChoice(D), RestChoice(D, seed=RECALL_DIRECTION_SEED)
    assert abs(float(torch.dot(rest.direction, recall.direction))) < 0.5
    assert torch.equal(RestChoice(D).direction, rest.direction), "rest's own direction is unchanged"


def test_recalls_higher_bar_is_pressed_by_chance_far_less_than_rests():
    choice = RestChoice(D, width=6.0, seed=RECALL_DIRECTION_SEED)
    generator = torch.Generator().manual_seed(3)
    nothing = torch.zeros(D)
    for _ in range(4000):
        choice.read(torch.randn(D, generator=generator), nothing)
    assert choice.chosen_count / choice.read_count < 0.002

