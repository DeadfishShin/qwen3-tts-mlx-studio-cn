from ui.components import format_table_md


def test_format_table_md_rows():
    md = format_table_md(["A", "B"], [["1", "2"], ["3", "4"]], "*empty*")
    lines = md.split("\n")
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"
    assert lines[3] == "| 3 | 4 |"


def test_format_table_md_escapes_pipes():
    md = format_table_md(["A"], [["x|y"]], "*empty*")
    assert "x\\|y" in md


def test_format_table_md_empty():
    assert format_table_md(["A"], [], "*empty*") == "*empty*"


def test_streaming_output_column_constructs():
    import gradio as gr
    from ui.components import build_output_column
    with gr.Blocks():
        out = build_output_column()
    assert out.audio.streaming is True
    assert out.stop is not None and out.result_state is not None
