"""
Workstream A: automated ground-truth query generator.
For each repo, AST-extract every function/class WITH a docstring.
Build an NL query from the first docstring sentence, stripping the identifier
name tokens so it can't trivially leak. Ground truth = identifier + file.
Keep only queries whose docstring sentence has >= 4 informative tokens.
"""
import ast, os, re, json, sys

REPOS = {
    'httpx':   '/tmp/httpx',
    'jinja2':  '/tmp/jinja2',
    'requests':'/tmp/requests',
    'flask':   '/tmp/flask',
    'click':   '/tmp/click',
    'rich':    '/tmp/rich',
}

STOPWORDS = {
    "find","the","a","an","where","how","that","which","all","into","of",
    "for","from","by","is","are","was","and","or","to","method","function",
    "class","module","locate","in","it","its","this","be","do","does","if",
    "up","at","on","out","as","with","via","per","no","not","but","their",
    "return","returns","given","use","used","using","when","then","else",
    "will","can","should","must","may","also","etc","e.g","i.e","you","your",
}

def split_identifier(name):
    """Split identifier into lowercase word tokens (no stemming here)."""
    n = name.lstrip('_')
    n = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', n)
    n = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', n)
    return [t.lower() for t in re.split(r'[_\s]+', n) if t]

def first_sentence(doc):
    doc = doc.strip()
    # take up to first period/newline-blank
    m = re.split(r'(?<=[.!?])\s|\n\n|\n', doc, maxsplit=1)
    return m[0].strip() if m else doc

def informative_tokens(text, exclude):
    raw = re.split(r'[\s\W]+', text.lower())
    return [t for t in raw if t and t not in STOPWORDS and t not in exclude and len(t) > 1]

def make_query(sentence, ident_tokens):
    """Strip identifier-name tokens from the sentence so it can't leak."""
    # build a regex that removes whole words equal to any ident token (case-insensitive)
    out = sentence
    for tok in sorted(set(ident_tokens), key=len, reverse=True):
        if len(tok) < 2:
            continue
        out = re.sub(r'\b' + re.escape(tok) + r'\b', '', out, flags=re.IGNORECASE)
    out = re.sub(r'\s+', ' ', out).strip(' .,:;-')
    return out

def extract(repo_name, repo_root):
    queries = []
    seen_names = set()
    for dp, _, files in os.walk(repo_root):
        # skip vendored/test dirs from query GENERATION? keep tests out of ground-truth
        for fn in files:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dp, fn)
            try:
                src = open(fp, errors='replace').read()
                tree = ast.parse(src, filename=fp)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                ident = node.name
                # skip dunders and trivial names
                if ident.startswith('__') and ident.endswith('__'):
                    continue
                ident_tokens = split_identifier(ident)
                sent = first_sentence(doc)
                if len(sent) < 10:
                    continue
                # informative tokens AFTER removing identifier tokens (so query can't leak)
                info = informative_tokens(sent, set(ident_tokens))
                if len(info) < 4:
                    continue
                query_text = make_query(sent, ident_tokens)
                # require the stripped query still has >= 4 informative tokens
                if len(informative_tokens(query_text, set(ident_tokens))) < 4:
                    continue
                rel_file = os.path.relpath(fp, repo_root)
                key = (ident, rel_file)
                if key in seen_names:
                    continue
                seen_names.add(key)
                queries.append({
                    'repo': repo_name,
                    'query': query_text,
                    'ground_truth': ident,
                    'gt_file': rel_file,
                    'gt_file_abs': fp,
                    'line': node.lineno,
                    'docstring_sentence': sent[:200],
                    'kind': type(node).__name__,
                })
    return queries

all_queries = []
per_repo = {}
for name, root in REPOS.items():
    qs = extract(name, root)
    per_repo[name] = len(qs)
    all_queries.extend(qs)

print('=== PER-REPO QUERY COUNTS ===')
for name, n in per_repo.items():
    print(f'  {name}: {n}')
print(f'  TOTAL: {len(all_queries)}')

# write jsonl
out_path = '/tmp/context_retrieval_queries.jsonl'
with open(out_path, 'w') as fh:
    for q in all_queries:
        fh.write(json.dumps(q) + '\n')
print(f'Wrote {out_path}')

# sample a few
print('\n=== SAMPLE QUERIES ===')
for q in all_queries[:5]:
    print(f"  [{q['repo']}] gt={q['ground_truth']} :: query={q['query'][:90]!r}")
