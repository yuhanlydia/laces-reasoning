import torch

from models.multichoice_reasoning import (
    MultipleChoiceFeatureRecord,
    MultipleChoiceResidualHead,
    format_multiple_choice_prompt,
    load_feature_record,
    recurrent_multichoice_objective,
    retarget_cosine_schedule,
    save_feature_record,
    stratified_three_way_split,
)
from models.recurrent_latent_reasoner import ReasoningTrace


def test_prompt_labels_all_options_and_ends_at_answer_slot():
    evidence, query = format_multiple_choice_prompt(
        "What is 2+2?", ["3", "4", "5"], category="math"
    )

    assert "Category: math" in evidence
    assert "(A) 3" in evidence
    assert "(B) 4" in evidence
    assert "(C) 5" in evidence
    assert "Answer:" not in evidence
    assert query == "Answer:"


def test_stratified_split_is_deterministic_disjoint_and_complete():
    categories = ["math"] * 10 + ["law"] * 10
    first = stratified_three_way_split(categories, seed=7)
    second = stratified_three_way_split(categories, seed=7)

    assert first == second
    train, validation, test = map(set, first)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(20))
    assert [sum(categories[i] == "math" for i in part) for part in first] == [8, 1, 1]


def test_multichoice_feature_round_trip(tmp_path):
    record = MultipleChoiceFeatureRecord(
        example_id="42",
        category="math",
        evidence=torch.randn(5, 8),
        query=torch.randn(2, 8),
        base_state_features=torch.randn(12),
        base_choice_logits=torch.tensor([-2.0, -1.0, -3.0]),
        num_choices=3,
        label=1,
    )
    path = tmp_path / "record.pt"
    save_feature_record(path, record)

    loaded = load_feature_record(path)
    assert loaded.example_id == "42"
    assert loaded.category == "math"
    assert loaded.num_choices == 3
    assert loaded.label == 1
    assert torch.equal(loaded.base_choice_logits, record.base_choice_logits)


def test_residual_head_starts_at_frozen_backbone_logits_and_masks_padding():
    head = MultipleChoiceResidualHead(state_dim=12, hidden_dim=16, max_choices=10)
    correction = torch.randn(2, 12)
    base = torch.tensor([[1.0, 3.0, 2.0, 99.0], [2.0, 1.0, 99.0, 99.0]])
    counts = torch.tensor([3, 2])

    logits = head(correction, base, counts)

    assert torch.equal(logits[0, :3], base[0, :3])
    assert torch.equal(logits[1, :2], base[1, :2])
    assert torch.isneginf(logits[0, 3])
    assert torch.isneginf(logits[1, 2:]).all()


def test_recurrent_objective_backpropagates_through_writer_correction():
    head = MultipleChoiceResidualHead(state_dim=12, hidden_dim=16, max_choices=10)
    correction = torch.randn(1, 12, requires_grad=True)
    trace = ReasoningTrace(
        latents=[torch.zeros(1, 4), torch.ones(1, 4)],
        contexts=[],
        cumulative_states=[],
        pooled_corrections=[torch.zeros(1, 12), correction],
        attention_weights=[],
    )
    loss, metrics = recurrent_multichoice_objective(
        head,
        trace,
        base_choice_logits=torch.zeros(1, 10),
        choice_counts=torch.tensor([4]),
        labels=torch.tensor([2]),
        depths=[1],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert correction.grad is not None
    assert torch.isfinite(correction.grad).all()
    assert metrics["supervised_depths"] == 1


def test_resume_retargets_cosine_schedule_to_new_total_step_budget():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50_000)

    retarget_cosine_schedule(scheduler, total_steps=500_000, grad_accum=4)

    assert scheduler.T_max == 125_000
