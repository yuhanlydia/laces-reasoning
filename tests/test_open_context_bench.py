import json

from scripts.eval.run_open_context_bench import (
    babilong_score,
    load_babilong_rows,
    load_longbench_rows,
    normalized_exact_match,
    qa_f1,
    rouge_l_f1,
    ruler_score,
)


def test_longbench_adapter_uses_official_prompt_and_answers(tmp_path):
    data = tmp_path / "hotpotqa.jsonl"
    data.write_text(json.dumps({"input": "Who won?", "context": "Ada won.", "answers": ["Ada"], "_id": "x"}) + "\n")
    prompts = {"hotpotqa": "Context: {context}\nQuestion: {input}\nAnswer:"}

    rows = load_longbench_rows(data, "hotpotqa", prompts)

    assert rows == [{"id": "x", "prompt": "Context: Ada won.\nQuestion: Who won?\nAnswer:", "answers": ["Ada"]}]


def test_babilong_adapter_builds_open_answer_prompt(tmp_path):
    data = tmp_path / "qa2.json"
    data.write_text(json.dumps([{"input": "Mary went home.", "question": "Where is Mary? ", "target": "home"}]))

    rows = load_babilong_rows(data)

    assert rows == [{
        "id": 0,
        "prompt": "Mary went home.\nQuestion: Where is Mary?\nAnswer:",
        "question": "Where is Mary?",
        "answers": ["home"],
    }]


def test_open_text_metrics_are_answer_normalized():
    assert normalized_exact_match("The Bathroom.", "bathroom") == 1.0
    assert qa_f1("Miller v California", "Miller v. California") == 1.0
    assert 0.0 < rouge_l_f1("alpha beta gamma", "alpha gamma") < 1.0


def test_babilong_score_matches_official_label_filtering():
    assert babilong_score("qa1", "Mary is in the bathroom. Question: Where is John?", "bathroom", "Where is Mary?") == 1.0
    assert babilong_score("qa1", "bathroom and bedroom", "bathroom", "Where is Mary?") == 0.0


def test_ruler_score_uses_official_any_and_all_substring_rules():
    assert ruler_score("qa_1", "The answer is Ada.", ["Ada", "Grace"]) == 1.0
    assert ruler_score("vt", "x and y", ["x", "y"]) == 1.0
    assert ruler_score("vt", "x alone", ["x", "y"]) == 0.5
