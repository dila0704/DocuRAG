import query_rewriter


def test_generate_hypothetical_answer_calls_client(fake_llm_client):
    client = fake_llm_client(["Bu bir laptop talep formudur, XYZ departmani icin."])
    result = query_rewriter.generate_hypothetical_answer("laptop talebi", client=client)

    assert result == "Bu bir laptop talep formudur, XYZ departmani icin."
    assert len(client.calls) == 1
    assert client.calls[0]["user_message"] == "laptop talebi"


def test_generate_hypothetical_answer_uses_higher_temperature_by_default(fake_llm_client):
    client = fake_llm_client(["x"])
    query_rewriter.generate_hypothetical_answer("sorgu", client=client)
    assert client.calls[0]["temperature"] == query_rewriter.DEFAULT_HYDE_TEMPERATURE
    assert query_rewriter.DEFAULT_HYDE_TEMPERATURE > 0.0


def test_condense_conversation_returns_follow_up_unchanged_when_no_history(fake_llm_client):
    client = fake_llm_client([])
    result = query_rewriter.condense_conversation([], "laptop talebi", client=client)
    assert result == "laptop talebi"
    assert client.calls == []  # gecmis bossa LLM'e hic gidilmemeli


def test_condense_conversation_calls_client_with_history_when_present(fake_llm_client):
    client = fake_llm_client(["laptop talebi belgesinin tarihi"])
    history = [{"query": "laptop talebi", "answer_summary": "Dila Alpay tarafindan olusturuldu."}]

    result = query_rewriter.condense_conversation(history, "peki tarihi neydi?", client=client)

    assert result == "laptop talebi belgesinin tarihi"
    assert len(client.calls) == 1
    assert "laptop talebi" in client.calls[0]["user_message"]
    assert "peki tarihi neydi?" in client.calls[0]["user_message"]


def test_condense_conversation_default_temperature_is_zero(fake_llm_client):
    client = fake_llm_client(["x"])
    query_rewriter.condense_conversation([{"query": "a", "answer_summary": "b"}], "c", client=client)
    assert client.calls[0]["temperature"] == 0.0
