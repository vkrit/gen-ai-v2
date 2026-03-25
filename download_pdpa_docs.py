from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import httpx


@dataclass(frozen=True)
class SourceDocument:
    slug: str
    title: str
    url: str


PDPA_SOURCES = [
    SourceDocument(
        slug="gppc_home",
        title="GPPC Home",
        url="https://gppc.pdpc.or.th/",
    ),
    SourceDocument(
        slug="pdpa_courses_pre_form",
        title="PDPA Courses Pre-Form",
        url="https://gppc.pdpc.or.th/pre-form-to-pdpa-courses/",
    ),
    SourceDocument(
        slug="pdpa_general_public_course",
        title="PDPA Courses for General Public",
        url="https://gppc.pdpc.or.th/pdpa-courses-for-general-public/",
    ),
    SourceDocument(
        slug="privacy_policy",
        title="GPPC Privacy Policy",
        url="https://gppc.pdpc.or.th/privacy-policy/",
    ),
    SourceDocument(
        slug="dpo_operational_training_entry",
        title="Operational Training for Data Protection Officer",
        url="https://gppc.pdpc.or.th/entry-form-to-operational-training-for-data-protection-officer-dpo/",
    ),
]


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "title",
        "tr",
        "ul",
        "ol",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = unescape("".join(self._parts))
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def fetch_page_text(client: httpx.Client, url: str) -> str:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()

    extractor = HTMLTextExtractor()
    extractor.feed(response.text)
    text = extractor.get_text()
    if not text:
        raise ValueError(f"No text extracted from {url}")
    return text


def save_document(output_dir: Path, doc: SourceDocument, text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{doc.slug}.txt"
    destination.write_text(
        f"Title: {doc.title}\nSource: {doc.url}\n\n{text}\n",
        encoding="utf-8",
    )
    return destination


def download_pdpa_docs(output_dir: Path) -> None:
    with httpx.Client(timeout=30.0, headers={"User-Agent": "Mozilla/5.0 PDPA downloader"}) as client:
        for doc in PDPA_SOURCES:
            try:
                text = fetch_page_text(client, doc.url)
                saved_path = save_document(output_dir, doc, text)
                print(f"Saved {saved_path}")
            except Exception as exc:
                print(f"Failed to download {doc.url}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download official PDPA pages into txt files")
    parser.add_argument(
        "--output-dir",
        default="docs/pdpa",
        help="Folder to save txt files into (default: docs/pdpa)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    download_pdpa_docs(output_dir)


if __name__ == "__main__":
    main()
