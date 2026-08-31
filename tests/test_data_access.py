import json

import data_access


def test_load_usage_log_returns_empty_dataframe_when_file_missing(tmp_path):
    df = data_access.load_usage_log(str(tmp_path / "does_not_exist.jsonl"))
    assert df.empty
    assert "cost_usd" in df.columns


def test_load_usage_log_parses_existing_records(tmp_path):
    path = tmp_path / "usage_log.jsonl"
    records = [
        {"timestamp": 1.0, "provider": "AnthropicClient", "model_name": "claude-sonnet-5",
         "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001, "duration_s": 1.2},
        {"timestamp": 2.0, "provider": "OpenAIClient", "model_name": "gpt_4",
         "input_tokens": 200, "output_tokens": 80, "cost_usd": None, "duration_s": 0.8},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    df = data_access.load_usage_log(str(path))
    assert len(df) == 2
    assert list(df["provider"]) == ["AnthropicClient", "OpenAIClient"]
    assert df.iloc[0]["input_tokens"] == 100
