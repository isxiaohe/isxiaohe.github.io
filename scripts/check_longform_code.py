"""Check that the pruning lecture renders highlighted code without line wrappers."""
from html.parser import HTMLParser
from pathlib import Path

ARTICLE = Path("public/notes/efficient-ml-part-1-pruning/index.html")


class LectureCodeParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_code_block = False
        self.blocks = []
        self.colors = []
        self._parts = []
        self._tags = []
        self._colors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "pre" and "lecture-code" in classes:
            assert not self.in_code_block
            self.in_code_block = True
            self._parts = []
            self._colors = set()
        elif self.in_code_block:
            if self._tags == ["code"]:
                assert tag == "span", f"Unexpected markup inside code: {tag}"
                styles = dict(
                    rule.split(":", 1)
                    for rule in attrs.get("style", "").split(";")
                    if ":" in rule
                )
                assert "display" not in styles, "Syntax tokens must stay inline"
                if "color" in styles:
                    self._colors.add(styles["color"])
            else:
                assert not self._tags and tag == "code", f"Unexpected markup inside code block: {tag}"
            self._tags.append(tag)

    def handle_data(self, data):
        if self.in_code_block:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if not self.in_code_block:
            return
        if tag == "pre":
            assert not self._tags, f"Unclosed code markup: {self._tags}"
            self.blocks.append("".join(self._parts))
            self.colors.append(self._colors)
            self.in_code_block = False
        else:
            assert self._tags and self._tags[-1] == tag, f"Mismatched closing tag: {tag}"
            self._tags.pop()


def main():
    html = ARTICLE.read_text(encoding="utf-8")
    parser = LectureCodeParser()
    parser.feed(html)

    assert len(parser.blocks) == 7, (
        f"Expected seven code examples, found {len(parser.blocks)}"
    )
    assert all(len(colors) > 1 for colors in parser.colors), (
        "Expected multiple syntax colors in each Python block"
    )
    assert any("# Each four-element group keeps" in b for b in parser.blocks), (
        "Python comments are missing from code blocks"
    )
    assert any("assert torch.cuda.is_available()" in b for b in parser.blocks)
    assert any("input_scale = X.square().sum(dim=0).sqrt()" in b for b in parser.blocks)
    assert 'class="score-catalog"' in html or "score-catalog" in html
    print(f"Long-form code rendering verified: {len(parser.blocks)} highlighted blocks")


if __name__ == "__main__":
    main()
