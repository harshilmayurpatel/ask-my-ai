"""Build the FAISS index from the Markdown files in knowledge/.

    python ingest.py

Each "## " section becomes one chunk (split further only if it is long),
with its file and section titles prepended so retrieval sees the context.
Sections whose body still starts with "TODO" are skipped.
"""
import re

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app import config


def load_chunks() -> list[Document]:
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc"), ("##", "section")], strip_headers=True
    )
    long_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks: list[Document] = []
    skipped = 0

    for path in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        for part in by_heading.split_text(text):
            body = part.page_content.strip()
            if not body:
                continue
            if body.upper().startswith("TODO"):
                skipped += 1
                continue
            doc, section = part.metadata.get("doc", path.stem), part.metadata.get("section", "")
            title = f"{doc} - {section}" if section else doc
            for piece in long_splitter.split_text(body):
                chunks.append(
                    Document(
                        page_content=f"{title}\n{piece}",
                        metadata={"source": path.name, "doc": doc, "section": section or doc},
                    )
                )

    print(f"{len(chunks)} chunks from {config.KNOWLEDGE_DIR.name}/ ({skipped} TODO sections skipped)")
    return chunks


def embeddings() -> FastEmbedEmbeddings:
    config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return FastEmbedEmbeddings(model_name=config.EMBED_MODEL, cache_dir=str(config.MODEL_CACHE))


def main() -> None:
    store = FAISS.from_documents(load_chunks(), embeddings())
    store.save_local(str(config.INDEX_DIR))
    print(f"Saved index to {config.INDEX_DIR.name}/")


if __name__ == "__main__":
    main()
