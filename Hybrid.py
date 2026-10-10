import re, sys
import numpy as np
import tree_sitter_python as tsp
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

src = open(sys.argv[1], "rb").read()
root = Parser(Language(tsp.language())).parse(src).root_node

def unwrap(n):
    return n.child_by_field_name("definition") if n.type == "decorated_definition" else n

def ast_chunks(root):
    out = []
    for node in root.children:
        inner = unwrap(node)
        if inner.type == "function_definition":
            out.append((inner.child_by_field_name("name").text.decode(), node))
        elif inner.type == "class_definition":
            cname = inner.child_by_field_name("name").text.decode()
            for m in inner.child_by_field_name("body").children:
                mi = unwrap(m)
                if mi.type == "function_definition":
                    out.append((f"{cname}.{mi.child_by_field_name('name').text.decode()}", m))
    return out

chunks = [(f"{n}@{node.start_point[0]+1}", node.text.decode()) for n, node in ast_chunks(root)]

def tokens(text):
    """lowercase, keep whole identifier AND its camelCase/snake_case parts"""
    out = []
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        out.append(w.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w.replace("_", " "))
        out += [p.lower() for p in parts]
    return out

bm25 = BM25Okapi([tokens(t) for _, t in chunks])
model = SentenceTransformer("all-MiniLM-L6-v2")
V = model.encode([t for _, t in chunks], normalize_embeddings=True)

def rrf(*rankings, k=60):
    score = {}
    for r in rankings:
        for rank, i in enumerate(r, start=1):
            score[i] = score.get(i, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

for q in sys.argv[2:]:
    b = list(np.argsort(-bm25.get_scores(tokens(q))))
    v = list(np.argsort(-(V @ model.encode(q, normalize_embeddings=True))))
    h = rrf(b, v)

    # print(f"\nQ: {q}")
    # print(f"{'BM25':32}{'VECTOR':32}HYBRID")
    # for j in range(5):
    #     print(f"{chunks[b[j]][0]:32}{chunks[v[j]][0]:32}{chunks[h[j]][0]}")

    cand = h
    sc = reranker.predict([(q, chunks[i][1]) for i in cand])
    rr = [cand[j] for j in np.argsort(-sc)]

    print(f"\nQ: {q}")
    print(f"{'BM25':40}{'VECTOR':40}{'HYBRID':40}RERANKED")
    for j in range(5):
        print(f"{chunks[b[j]][0]:40}{chunks[v[j]][0]:40}{chunks[h[j]][0]:40}{chunks[rr[j]][0]}")