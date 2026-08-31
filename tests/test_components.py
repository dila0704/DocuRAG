from collections import Counter
from datetime import datetime, timedelta, timezone

import components


def test_confidence_stamp_high_threshold_inclusive():
    assert 'stamp-high' in components.render_confidence_stamp(0.85)
    assert 'stamp-high' in components.render_confidence_stamp(0.95)


def test_confidence_stamp_mid_range():
    assert 'stamp-mid' in components.render_confidence_stamp(0.6)
    assert 'stamp-mid' in components.render_confidence_stamp(0.849)


def test_confidence_stamp_low_range_includes_review_label():
    html = components.render_confidence_stamp(0.59)
    assert 'stamp-low' in html
    assert 'İNCELEME BEKLİYOR' in html


def test_confidence_stamp_none_score_is_low():
    html = components.render_confidence_stamp(None)
    assert 'stamp-low' in html
    assert '—' in html


def test_category_badge_uses_correct_color_variable():
    html = components.render_category_badge("fatura")
    assert "var(--cat-fatura)" in html
    assert "fatura" in html


def test_category_badge_unknown_category_falls_back_to_diger():
    html = components.render_category_badge("bilinmeyen")
    assert "var(--cat-diger)" in html


def _chunk(text, source_doc, score, **extra):
    return {"text": text, "source_doc": source_doc, "score": score, **extra}


def test_format_chunk_score_cosine_shows_percentage():
    label = components.format_chunk_score({"_citation_index": 1, "score": 0.873})
    assert label == "[1] %87"


def test_format_chunk_score_rrf_shows_raw_score_not_percentage():
    label = components.format_chunk_score({"_citation_index": 2, "score": 0.0325, "score_type": "rrf"})
    assert label == "[2] RRF 0.033"


def test_group_results_by_document_groups_same_source():
    chunks = [
        _chunk("a1", "a.png", 0.9, _citation_index=1),
        _chunk("b1", "b.png", 0.5, _citation_index=2),
        _chunk("a2", "a.png", 0.7, _citation_index=3),
    ]
    groups = components.group_results_by_document(chunks)
    assert len(groups) == 2
    a_group = next(g for g in groups if g["source_doc"] == "a.png")
    assert len(a_group["chunks"]) == 2
    assert a_group["best_score"] == 0.9
    assert [c["_citation_index"] for c in a_group["chunks"]] == [1, 3]


def test_group_results_by_document_sorted_by_best_score_desc():
    chunks = [_chunk("x", "low.png", 0.3), _chunk("y", "high.png", 0.9)]
    groups = components.group_results_by_document(chunks)
    assert [g["source_doc"] for g in groups] == ["high.png", "low.png"]


def test_group_results_empty_list():
    assert components.group_results_by_document([]) == []


def test_highlight_terms_wraps_matching_word_case_insensitive():
    html = components.highlight_terms("Yeni Laptop Talebi", "laptop talebi")
    assert '<span class="mark">Laptop</span>' in html
    assert '<span class="mark">Talebi</span>' in html


def test_highlight_terms_escapes_html_in_source_text():
    html = components.highlight_terms("<b>fatura</b> tutari", "fatura")
    assert "&lt;b&gt;" in html
    assert '<span class="mark">fatura</span>' in html


def test_highlight_terms_ignores_short_query_words():
    html = components.highlight_terms("bu bir metin", "bu")
    assert "mark" not in html


def test_render_class_distribution_donut_empty_counter():
    html = components.render_class_distribution_donut(Counter())
    assert "Henüz" in html


def test_render_class_distribution_donut_percentages_sum_reasonable():
    counter = Counter({"fatura": 1, "talep formu": 1})
    html = components.render_class_distribution_donut(counter)
    assert "%50" in html
    assert "Fatura" in html
    assert "Talep formu" in html


def test_render_document_graph_svg_empty_graph_shows_message():
    import networkx as nx
    html_out = components.render_document_graph_svg(nx.Graph())
    assert "Henüz" in html_out


def test_render_document_graph_svg_renders_node_and_edge():
    import networkx as nx
    graph = nx.Graph()
    graph.add_edge("a.png", "b.png", weight=2, shared=["Dila Alpay"])
    html_out = components.render_document_graph_svg(graph)
    assert html_out.count("<circle") == 2
    assert "<line" in html_out
    assert "a.png" in html_out and "b.png" in html_out


def test_render_latency_bars_empty_shows_message():
    assert "Henüz" in components.render_latency_bars({})


def test_render_latency_bars_scales_widest_bar_to_100_percent():
    html_out = components.render_latency_bars({"AnthropicClient": 2.0, "LocalHFClient": 1.0})
    assert "AnthropicClient" in html_out and "LocalHFClient" in html_out
    assert "width:100.0%" in html_out
    assert "width:50.0%" in html_out


class _FakePlaceholder:
    def __init__(self):
        self.calls: list[str] = []

    def markdown(self, body, **kwargs):
        self.calls.append(body)


def test_stream_sentences_reveals_sentences_progressively():
    placeholder = _FakePlaceholder()
    components.stream_sentences(["Birinci.", "İkinci.", "Üçüncü."], placeholder, delay=0)

    assert len(placeholder.calls) == 3
    assert placeholder.calls[0] == "Birinci."
    assert placeholder.calls[1] == "Birinci. İkinci."
    assert placeholder.calls[2] == "Birinci. İkinci. Üçüncü."


def test_stream_sentences_applies_wrap_function():
    placeholder = _FakePlaceholder()
    components.stream_sentences(["x"], placeholder, wrap=lambda body: f"<div>{body}</div>", delay=0)
    assert placeholder.calls == ["<div>x</div>"]


def test_stream_sentences_empty_list_does_not_call_placeholder():
    placeholder = _FakePlaceholder()
    components.stream_sentences([], placeholder, delay=0)
    assert placeholder.calls == []


def test_search_results_to_dataframe_has_expected_columns_and_rows():
    results = [
        {"source_doc": "a.png", "score": 0.9, "score_type": "cosine", "siniflar": ["fatura"], "guven": 0.8, "text": "metin"},
    ]
    df = components.search_results_to_dataframe(results)
    assert list(df.columns) == ["belge", "skor", "skor_turu", "siniflar", "guven", "metin"]
    assert len(df) == 1
    assert df.iloc[0]["belge"] == "a.png"


def test_search_results_to_dataframe_empty_list():
    df = components.search_results_to_dataframe([])
    assert df.empty


def test_render_printable_report_html_includes_print_color_adjust_and_documents():
    documents = {"a.png": {"siniflar": ["fatura"], "guven": 0.9, "ingested_at": None}}
    html_out = components.render_printable_report_html(documents, Counter({"fatura": 1}))
    assert "-webkit-print-color-adjust: exact" in html_out
    assert "a.png" in html_out
    assert "fatura" in html_out.lower()


def test_render_printable_report_html_includes_usage_summary_when_given():
    html_out = components.render_printable_report_html({}, Counter(), {"total_calls": 5, "total_cost": 1.2345, "avg_duration": 0.5})
    assert "Model Kullanımı" in html_out
    assert "1.2345" in html_out


def test_format_relative_time_missing_returns_dash():
    assert components.format_relative_time(None) == "-"
    assert components.format_relative_time("not-a-date") == "-"


def test_format_relative_time_minutes_ago():
    ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert components.format_relative_time(ts) == "5 dk önce"


def test_format_relative_time_hours_ago():
    ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assert components.format_relative_time(ts) == "2 sa önce"


def test_format_relative_time_just_now():
    ts = datetime.now(timezone.utc).isoformat()
    assert components.format_relative_time(ts) == "az önce"
