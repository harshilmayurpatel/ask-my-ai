"""Accuracy test for Ask my AI.

    python -m eval.run_eval            # retrieval check always; answer check if GROQ_API_KEY is set
    python -m eval.run_eval --url https://ask-my-ai.onrender.com   # test the deployed API instead

Retrieval: is the expected knowledge section among the top-k chunks?
Answers:   does the generated answer contain the expected facts?
"""
import argparse
import json
import sys
from pathlib import Path

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))


def norm(text: str) -> str:
    """Compare text the way a reader sees it: models sometimes emit narrow
    no-break spaces or non-breaking hyphens that look identical on screen."""
    for ch in (" ", " ", " ", " "):
        text = text.replace(ch, " ")
    for ch in ("‐", "‑", "‒", "–"):
        text = text.replace(ch, "-")
    return " ".join(text.lower().split())


def read_stream(text: str) -> tuple[str, list[str]]:
    answer, sources = "", []
    for line in text.splitlines():
        if line.startswith("data:"):
            ev = json.loads(line[5:])
            answer += ev.get("t", "")
            sources = ev.get("sources", sources)
    return answer, sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="deployed API base URL (default: run in-process)")
    args = ap.parse_args()

    if args.url:
        import httpx

        import time

        def ask(q):
            # The live API allows 8 questions a minute per visitor: pace the
            # test, and wait out a 429 instead of failing
            time.sleep(8)
            for _ in range(3):
                r = httpx.post(args.url.rstrip("/") + "/chat", json={"question": q}, timeout=90)
                if r.status_code != 429:
                    break
                time.sleep(61)
            r.raise_for_status()
            return read_stream(r.text)
        check_answers = True
    else:
        from app import config, rag

        def ask(q):
            docs = rag.retrieve(q)
            import asyncio

            async def collect():
                return "".join([p async for p in rag.stream_answer(q, docs, [])])
            return asyncio.run(collect()), rag.sources(docs)
        check_answers = bool(config.GROQ_API_KEY)

    r_ok = r_total = a_ok = 0
    for item in QUESTIONS:
        answer, sources = ask(item["q"])
        line = []
        if item["section"]:
            r_total += 1
            hit = item["section"] in sources
            r_ok += hit
            line.append("retrieval " + ("ok " if hit else "MISS"))
        if check_answers:
            good = all(norm(e) in norm(answer) for e in item["expect"])
            a_ok += good
            line.append("answer " + ("ok " if good else "MISS"))
        print(f"[{' | '.join(line)}] {item['q']}")
        if check_answers and not good:
            print(f"      -> {answer[:200]!r}")

    print(f"\nRetrieval: {r_ok}/{r_total} questions found the right section")
    if check_answers:
        print(f"Answers:   {a_ok}/{len(QUESTIONS)} contained the expected facts")
    else:
        print("Answers:   skipped (set GROQ_API_KEY in .env to check generated answers)")
    return 0 if r_ok == r_total else 1


if __name__ == "__main__":
    sys.exit(main())
