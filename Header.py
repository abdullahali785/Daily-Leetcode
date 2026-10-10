import os, sys
import numpy as np
import tree_sitter_python as tsp
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer

parser = Parser(Language(tsp.language()))

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

chunks = []  # (label, bare_text, headed_text)
for dirpath, _, files in os.walk(sys.argv[1]):
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        root = parser.parse(open(path, "rb").read()).root_node
        for name, node in ast_chunks(root):
            code = node.text.decode()
            header = f"# file: {path}\n# symbol: {name}\n"
            chunks.append((f"{path}:{name}@{node.start_point[0]+1}", code, header + code))

model = SentenceTransformer("all-MiniLM-L6-v2")
bare = model.encode([c[1] for c in chunks], normalize_embeddings=True)
head = model.encode([c[2] for c in chunks], normalize_embeddings=True)
print(f"{len(chunks)} chunks")

for q in sys.argv[2:]:
    qv = model.encode(q, normalize_embeddings=True)
    b = np.argsort(-(bare @ qv))[:5]
    h = np.argsort(-(head @ qv))[:5]
    print(f"\nQ: {q}\n{'WITHOUT HEADER':62}WITH HEADER")
    for j in range(5):
        print(f"{chunks[b[j]][0]:62}{chunks[h[j]][0]}")