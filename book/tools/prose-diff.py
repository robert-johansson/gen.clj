"""Prose changes (Markdown outside code blocks) to book chapters between two commits.

usage: python3 -I prose-diff.py BASE HEAD [glob ...]   (run in the repository)
Prints, per chapter, each changed paragraph as - old / + new. Code blocks
(``` fences) are dropped first, so code-only edits do not show.
"""
import difflib
import fnmatch
import subprocess
import sys


def show(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def prose_paragraphs(text):
    out, para, fence = [], [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
            if para:
                out.append(" ".join(para)); para = []
            continue
        if fence or line.startswith("<!-- code:"):
            continue
        if line.strip() == "":
            if para:
                out.append(" ".join(para)); para = []
        else:
            para.append(line.strip())
    if para:
        out.append(" ".join(para))
    return out


def main():
    base, head, globs = sys.argv[1], sys.argv[2], sys.argv[3:] or ["book/0[234]*.md"]
    files = subprocess.run(["git", "diff", "--name-only", base, head, "--", "book/"],
                           capture_output=True, text=True).stdout.split()
    files = [f for f in files if f.endswith(".md") and any(fnmatch.fnmatch(f, g) for g in globs)]
    total = 0
    for f in sorted(files):
        old, new = show(base, f), show(head, f)
        if old is None or new is None:
            continue
        a, b = prose_paragraphs(old), prose_paragraphs(new)
        changes = [(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2
                   in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]
        if not changes:
            continue
        print(f"\n## {f} ({len(changes)} changed passages)")
        for tag, olds, news in changes:
            total += 1
            for p in olds:
                print(f"  - {p}")
            for p in news:
                print(f"  + {p}")
            print()
    print(f"\n{total} changed passages in {len(files)} files with diffs")


main()
