#!/usr/bin/env python3
# One-shot: download C++ rows from The Stack → extract masked functions → JSONL
# Requires:
#   pip install datasets tree-sitter==0.20.4 tree-sitter-languages==1.10.2
#   pip install python-dotenv huggingface_hub

from pathlib import Path
import json, re, os, sys
from tree_sitter import Node
from tree_sitter_languages import get_parser
from datasets import load_dataset
from tqdm import tqdm

# --- HF auth from .env ---
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()  # loads HF_TOKEN from .env in current working dir
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env. Add a line like: HF_TOKEN=hf_XXXXXXXX")
login(token=HF_TOKEN)

# ===================== CONFIG =====================
CACHE_DIR      = "./the_stack_cpp"    # HF datasets cache on disk
SAVE_CPP_DIR   = Path("cpp_samples")  # where to save a few raw .cpp files (optional)
SAVE_CPP_N     = 20                   # save first N .cpp files (set 0 to skip)
EXAMPLE_JSONL  = Path("example_output.jsonl")  # small example for GitHub
EXAMPLE_N      = 5                    # number of examples to save
OUTPUT_JSONL   = Path("data/data_cpp.jsonl")
MAX_FUNCS      = 3000                  # stop after writing this many functions
DEBUG          = True                 # print small progress
# =================================================

# Init parser once
parser = get_parser("cpp")

# ---------- Tree-sitter helpers ----------
def node_text(src: bytes, n: Node) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

def walk(n: Node):
    stack=[n]
    while stack:
        cur=stack.pop(); yield cur
        stack.extend(reversed(cur.children))

def is_function_definition(n: Node) -> bool:
    return n.type == "function_definition"

def get_func_name(func: Node, src: bytes) -> str:
    decl = func.child_by_field_name("declarator")
    if not decl: return ""
    name = ""
    for n in walk(decl):
        if n.type == "identifier":
            name = node_text(src, n)
    return name

def collect_param_names(func: Node, src: bytes) -> list[str]:
    decl = func.child_by_field_name("declarator")
    if not decl: return []
    names=set()
    for n in walk(decl):
        if n.type == "parameter_declaration":
            for m in walk(n):
                if m.type == "identifier":
                    nm = node_text(src, m)
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", nm):
                        names.add(nm)
    return sorted(names)

DECL_CTX = {
    "declaration","simple_declaration","structured_binding_declaration","init_declarator",
    "range_for_declaration","for_range_declaration","condition_declaration",
}

def ancestor_types(n: Node):
    out=[]; cur=n
    while cur: out.append(cur.type); cur=cur.parent
    return out

def collect_local_names(func: Node, src: bytes) -> list[str]:
    body = func.child_by_field_name("body")
    if not body: return []
    names=set()
    for n in walk(body):
        if n.type == "binding_identifier":
            nm = node_text(src, n)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", nm): names.add(nm)
            continue
        if n.type == "identifier":
            anc = ancestor_types(n)
            if any(t in DECL_CTX for t in anc):
                p = n.parent
                if p and p.type in {"type_identifier","primitive_type","qualified_identifier"}:
                    continue
                if "field_declaration" in anc:
                    continue
                nm = node_text(src, n)
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", nm):
                    names.add(nm)
    return sorted(names)

def mask_names(text: str, names: list[str]) -> tuple[str, dict]:
    mapping, masked = {}, text
    names = sorted(names, key=len, reverse=True)
    for i, nm in enumerate(names, 1):
        ph = f"<ID_{i}>"; mapping[ph] = nm
        pat = r"(?<![A-Za-z0-9_])" + re.escape(nm) + r"(?![A-Za-z0-9_])"
        masked = re.sub(pat, ph, masked)
    return masked, mapping

def extract_functions_from_source(source_text: str, file_tag: str):
    src = source_text.encode("utf-8", errors="ignore")
    tree = parser.parse(src)
    out=[]
    for fn in walk(tree.root_node):
        if not is_function_definition(fn): continue
        locals_ = collect_local_names(fn, src)
        params  = collect_param_names(fn, src)
        names   = sorted(set(locals_) | set(params))
        if not names:  # skip functions with no identifiers to rename
            continue
        func_txt = node_text(src, fn)
        masked, mapping = mask_names(func_txt, names)
        out.append({
            "file": file_tag,
            "func_name": get_func_name(fn, src) or "",
            "input_text": masked,
            "target_text": json.dumps(mapping, ensure_ascii=False),
        })
    return out
# ---------------------------------------

def main():
    # Load the C++ split (streaming avoids downloading everything)
    ds = load_dataset("bigcode/the-stack", data_dir="data/c++", split="train", streaming=True, cache_dir=CACHE_DIR)

    if SAVE_CPP_N > 0:
        SAVE_CPP_DIR.mkdir(parents=True, exist_ok=True)

    # Create data directory for output
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    saved_cpp = 0
    example_written = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fw, \
         EXAMPLE_JSONL.open("w", encoding="utf-8") as fw_example:
        # Use tqdm to show progress
        pbar = tqdm(desc="Processing C++ files", unit="funcs", total=MAX_FUNCS)
        for i, row in enumerate(ds):
            content = row.get("content")
            if not content:
                continue

            # Optionally save a few raw .cpp files (for debugging)
            if saved_cpp < SAVE_CPP_N:
                (SAVE_CPP_DIR / f"row_{i:06d}.cpp").write_text(content, encoding="utf-8")
                saved_cpp += 1

            # Extract functions from this row (only use first function per file)
            rows = extract_functions_from_source(content, file_tag=f"row_{i:06d}")
            if rows:  # Only process if we found at least one function
                r = rows[0]  # Take only the first function from this file
                fw.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)  # Update progress bar
                
                # Also write to example file (for GitHub)
                if example_written < EXAMPLE_N:
                    fw_example.write(json.dumps(r, ensure_ascii=False) + "\n")
                    example_written += 1
                
                if written >= MAX_FUNCS:
                    pbar.close()
                    if DEBUG:
                        print(f"[ok] wrote {written} functions → {OUTPUT_JSONL} (saved {saved_cpp} cpp samples, {example_written} examples)")
                    fw.flush()  # Ensure data is written
                    fw_example.flush()
                    os._exit(0)  # Force immediate exit without cleanup
            
        pbar.close()
    if DEBUG:
        print(f"[done] wrote {written} functions → {OUTPUT_JSONL} (saved {saved_cpp} cpp samples, {example_written} examples)")

if __name__ == "__main__":
    main()