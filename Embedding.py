import sys
import numpy as np
import tree_sitter_python as tsp
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer("all-MiniLM-L6-v2")   # ~90MB download on first run
vecs = model.encode([text for _, text in chunks])
print("vectors:", vecs.shape)

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

for q in sys.argv[2:]:
    qv = model.encode(q)
    top = sorted(((cos(qv, v), name) for (name, _), v in zip(chunks, vecs)), reverse=True)[:5]
    print(f"\nQ: {q}")
    for s, name in top:
        print(f"  {s:.3f}  {name}")