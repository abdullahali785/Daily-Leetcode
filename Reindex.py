import hashlib, json, os, sys
import tree_sitter_python as tsp
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"
CACHE = "embed_cache.json"
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

texts = []
for dirpath, _, files in os.walk(sys.argv[1]):
    for f in sorted(files):
        if f.endswith(".py"):
            path = os.path.join(dirpath, f)
            root = parser.parse(open(path, "rb").read()).root_node
            for name, node in ast_chunks(root):
                texts.append(f"# file: {path}\n# symbol: {name}\n" + node.text.decode())

by_hash = {hashlib.sha256((MODEL + "\0" + t).encode()).hexdigest(): t for t in texts}

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
missing = [h for h in by_hash if h not in cache]
if missing:
    model = SentenceTransformer(MODEL)          # only loaded if there is work to do
    vecs = model.encode([by_hash[h] for h in missing], normalize_embeddings=True)
    for h, v in zip(missing, vecs):
        cache[h] = v.tolist()
stale = [h for h in cache if h not in by_hash]
for h in stale:
    del cache[h]
json.dump(cache, open(CACHE, "w"))

print(f"{len(texts)} chunks | embedded {len(missing)} new | reused {len(by_hash) - len(missing)} | dropped {len(stale)} stale")