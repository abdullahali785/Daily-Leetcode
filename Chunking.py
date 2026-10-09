import sys
import tree_sitter_python as tsp
from tree_sitter import Language, Parser

SIZE, OVERLAP = 500, 50
STEP = SIZE - OVERLAP

src = open(sys.argv[1], "rb").read()

# --- fixed-size: slice every STEP bytes, SIZE long ---
fixed = [(i, src[i:i + SIZE]) for i in range(0, len(src), STEP)]

# --- AST-aware: one chunk per function / method ---
parser = Parser(Language(tsp.language()))
root = parser.parse(src).root_node

def unwrap(node):
    """decorated_definition -> the function/class inside it"""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition")
    
    return node

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
                    mname = mi.child_by_field_name("name").text.decode()
                    out.append((f"{cname}.{mname}", m))

    return out

print(f"FILE: {len(src)} bytes | FIXED: {len(fixed)} chunks\n")
print(f"{'symbol':40} {'lines':>9} {'bytes':>6}  fixed-size result")

for name, node in ast_chunks(root):
    s, e = node.start_byte, node.end_byte
    whole = [i for i, _ in fixed if i <= s and e <= i + SIZE]
    touched = [i for i, _ in fixed if s < i + SIZE and e > i]
    verdict = "kept whole" if whole else f"SPLIT across {len(touched)} chunks"

    print(f"{name:40} {node.start_point[0]+1:>4}-{node.end_point[0]+1:<4} {e-s:>6}  {verdict}")

# peek at where fixed chunks start/end mid-code
print("\nFIXED chunk boundaries (first/last 50 chars):")

for i, c in fixed[:6]:
    t = c.decode("utf8", "ignore")
    print(f"#{i//STEP}: {t[:50]!r} ... {t[-50:]!r}")