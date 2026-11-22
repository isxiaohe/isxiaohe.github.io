"""Validate the rendered pruning lecture's code blocks after the Hugo build.

The regression test protects against markdown parsing of Python comments and
against Chroma token wrappers reintroducing mobile font-size inflation.
"""
from html.parser import HTMLParser
from pathlib import Path

ARTICLE = Path("public/notes/efficient-ml-part-1-pruning/index.html")


class LectureCodeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_code_block = False
        self.blocks = []
        self.forbidden_tags = []
        self._parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "pre" and "lecture-code" in classes:
            assert not self.in_code_block
            self.in_code_block = True
            self._parts = []
        elif self.in_code_block and tag in {"span", "strong", "em", "h1", "h2", "h3", "h4"}:
            self.forbidden_tags.append(tag)

    def handle_data(self, data):
        if self.in_code_block:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "pre" and self.in_code_block:
            self.blocks.append("".join(self._parts))
            self.in_code_block = False


def main():
    html = ARTICLE.read_text(encoding="utf-8")
    parser = LectureCodeParser()
    parser.feed(html)

    assert len(parser.blocks) == 7, (
        f"Expected seven code examples, found {len(parser.blocks)}"
    )
    assert not parser.forbidden_tags, (
        f"Unexpected formatting inside code: {parser.forbidden_tags}"
    )
    assert any("# Each four-element group keeps" in b for b in parser.blocks), (
        "Python comments are missing from code blocks"
    )
    assert any("assert torch.cuda.is_available()" in b for b in parser.blocks)
    assert any("input_scale = X.square().sum(dim=0).sqrt()" in b for b in parser.blocks)
    assert 'class="score-catalog"' in html or "score-catalog" in html
    print(f"Long-form code rendering verified: {len(parser.blocks)} plain blocks")


if __name__ == "__main__":
    main()
