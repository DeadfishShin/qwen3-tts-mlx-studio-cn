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
