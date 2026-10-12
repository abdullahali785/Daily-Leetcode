import os, re, sys
import numpy as np
import tree_sitter_python as tsp
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

K = 5

# (query, relevant "path:symbol" labels). Write yours BEFORE running.
# "code:<label>@<line>" = use that chunk's code as the query (the chunk itself is excluded).
EVAL = [
    ("find two numbers that add up to a target", {"Solutions/easy.py:Solution.twoSum"}),
    ("twoSum", {"Solutions/easy.py:Solution.twoSum"}),
    ("maxDepth", {"Solutions/easy.py:Solution.maxDepth"}),
    ("max dept", {"Solutions/easy.py:Solution.maxDepth"}),
    ("check whether a string reads the same backwards",
        {"Solutions/easy.py:Solution.isPalindrome", "Solutions/easy.py:Solution.isPalindrome_pointer"}),
    ("code:Solutions/easy.py:Solution.twoSum@48", {"Solutions/easy.py:Solution.twoSum"}),
    ("code:Solutions/easy.py:Solution.isPalindrome@138",
        {"Solutions/easy.py:Solution.isPalindrome", "Solutions/easy.py:Solution.isPalindrome_pointer"}),
        ("does the array contain any repeated value", {"Solutions/easy.py:Solution.hasDuplicate"}),
    ("are two strings made of the same letters", {"Solutions/easy.py:Solution.isAnagram"}),
    ("check brackets are balanced and properly nested", {"Solutions/easy.py:Solution.isValid"}),
    ("reverse a singly linked list", {"Solutions/easy.py:Solution.reverseList"}),
    ("detect a loop in a linked list", {"Solutions/easy.py:Solution.hasCycle"}),
    ("maximum profit from buying and selling a stock once", {"Solutions/easy.py:Solution.maxProfit"}),
    ("traverse a binary tree level by level", {"Solutions/medium.py:Solution.levelOrder"}),
    ("longest substring without repeating characters", {"Solutions/medium.py:Solution.lengthOfLongestSubstring"}),
    ("mergeTwoLists", {"Solutions/easy.py:Solution.mergeTwoLists"}),
    ("levelorder", {"Solutions/medium.py:Solution.levelOrder"}),
    ("lowestCommonAncestor",
        {"Solutions/medium.py:Solution.lowestCommonAncestor1", "Solutions/medium.py:Solution.lowestCommonAncestor3"}),
    ("code:Solutions/medium.py:Solution.addTwoNumbers@659", {"Solutions/medium.py:Solution.addTwoNumbers_manual"}),
    ("code:Solutions/medium.py:Solution.lowestCommonAncestor1@819", {"Solutions/medium.py:Solution.lowestCommonAncestor3"}),
    ("code:Solutions/easy.py:Solution.search@214", {"Solutions/easy.py:Solution.search_helper"}),
]

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

chunks = []
for dirpath, _, files in os.walk(sys.argv[1]):
    for f in sorted(files):
        if f.endswith(".py"):
            path = os.path.join(dirpath, f)
            root = parser.parse(open(path, "rb").read()).root_node
            for name, node in ast_chunks(root):
                chunks.append({"label": f"{path}:{name}", "line": node.start_point[0] + 1,
                               "code": node.text.decode()})

def tokens(text):
    out = []
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        out.append(w.lower())
        out += [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w.replace("_", " "))]
    return out

def rrf(*rankings, k=60):
    score = {}
    for r in rankings:
        for rank, i in enumerate(r, start=1):
            score[i] = score.get(i, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)

bm25 = BM25Okapi([tokens(c["code"]) for c in chunks])
model = SentenceTransformer("all-MiniLM-L6-v2")
bare = model.encode([c["code"] for c in chunks], normalize_embeddings=True)
head = model.encode(
    [f"# file: {c['label'].split(':', 1)[0]}\n# symbol: {c['label'].split(':', 1)[1]}\n" + c["code"] for c in chunks],
    normalize_embeddings=True)

def run(q):
    exclude = None
    if q.startswith("code:"):
        exclude = next(i for i, c in enumerate(chunks) if f"{c['label']}@{c['line']}" == q[5:])
        q = chunks[exclude]["code"]
    qv = model.encode(q, normalize_embeddings=True)
    ranks = {
        "bm25": list(np.argsort(-bm25.get_scores(tokens(q)))),
        "vector": list(np.argsort(-(bare @ qv))),
        "vector+header": list(np.argsort(-(head @ qv))),
    }
    ranks["hybrid"] = rrf(ranks["bm25"], ranks["vector"])
    if exclude is not None:
        ranks = {n: [i for i in r if i != exclude] for n, r in ranks.items()}
    return ranks

names = ["bm25", "vector", "vector+header", "hybrid"]
tot = {n: [0.0, 0.0] for n in names}   # recall, mrr
print(f"{'query':48}" + "".join(f"{n:>15}" for n in names) + "   (rank of first hit)")
for q, rel in EVAL:
    ranks = run(q)
    row = ""
    for n in names:
        r = ranks[n]
        top = {chunks[i]["label"] for i in r[:K]}
        tot[n][0] += len(rel & top) / len(rel)
        first = next((j + 1 for j, i in enumerate(r) if chunks[i]["label"] in rel), None)
        tot[n][1] += 1 / first if first else 0
        row += f"{(first or '-'):>15}"
    print(f"{q[:46]:48}{row}")

N = len(EVAL)
print(f"\n{'':48}" + "".join(f"{n:>15}" for n in names))
print(f"{'recall@' + str(K):48}" + "".join(f"{tot[n][0]/N:>15.2f}" for n in names))
print(f"{'MRR':48}" + "".join(f"{tot[n][1]/N:>15.2f}" for n in names))