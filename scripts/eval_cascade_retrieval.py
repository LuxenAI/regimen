"""Confidence-gated lexical->neural cascade. Goal: capture neural-recoverable
lexical misses WITHOUT regressing the 96.6% lexical already gets."""
import ast, os, json, sys
import numpy as np
sys.path.insert(0,'/home/ubuntu/slmharness/src')
from openharness.orchestration.code_retriever import CodeRetriever, SymbolEntry
import tiktoken
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from collections import defaultdict

ENC=tiktoken.get_encoding("cl100k_base")
PORTER=PorterStemmer().stem
REPOS={'httpx':'/tmp/httpx','jinja2':'/tmp/jinja2','requests':'/tmp/requests',
       'flask':'/tmp/flask','click':'/tmp/click','rich':'/tmp/rich'}

def sig_of(node):
    try: return f"({', '.join(a.arg for a in node.args.args)})"
    except Exception: return "()"
def extract(root):
    syms=[]
    for dp,_,files in os.walk(root):
        for fn in files:
            if not fn.endswith('.py'): continue
            fp=os.path.join(dp,fn)
            try: tree=ast.parse(open(fp,errors='replace').read(),filename=fp)
            except SyntaxError: continue
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    doc=ast.get_docstring(node) or ""
                    sig=sig_of(node) if not isinstance(node,ast.ClassDef) else ""
                    syms.append({'name':node.name,'file':os.path.relpath(fp,root),
                        'kind':type(node).__name__,'doc':doc[:500],
                        'text':f"{node.name}{sig}\n{doc[:500]}".strip()})
    return syms

print('Loading neural model...',flush=True)
M=SentenceTransformer('jinaai/jina-embeddings-v2-base-code',trust_remote_code=True,device='cuda')
repo_syms={};lex={};neu_emb={}
for name,root in REPOS.items():
    syms=extract(root); repo_syms[name]=syms
    ents=[SymbolEntry(name=s['name'],file_path=s['file'],docstring=s['doc'],kind=s['kind']) for s in syms]
    lex[name]=CodeRetriever(stemmer=PORTER,splitter=lambda t:[t],test_penalty=1.0).fit(ents)
    neu_emb[name]=M.encode([s['text'] for s in syms],batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    print(f'  {name} indexed {len(syms)}',flush=True)

queries=[json.loads(l) for l in open('/home/ubuntu/slmharness/evals/context_retrieval_queries.jsonl')]
byrepo=defaultdict(list)
for q in queries: byrepo[q['repo']].append(q)

def find_rank(order_keys, gt, gtf, k=10):
    for r,(nm,f) in enumerate(order_keys[:k],1):
        if nm==gt and f==gtf: return r
    return None

# Precompute lexical and neural ranked key-lists per query
data=[]  # (q, lex_keys, lex_scores, neu_keys)
for name,qs in byrepo.items():
    qemb=M.encode([q['query'] for q in qs],batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    sims=qemb @ neu_emb[name].T
    syms=repo_syms[name]
    for qi,q in enumerate(qs):
        lres=lex[name].query(q['query'],top_k=20)
        lex_keys=[(r.entry.name,r.entry.file_path) for r in lres]
        lex_scores=[r.score for r in lres]
        norder=np.argsort(-sims[qi])[:20]
        neu_keys=[(syms[i]['name'],syms[i]['file']) for i in norder]
        data.append((q,lex_keys,lex_scores,neu_keys))

def lex_margin(scores):
    if len(scores)<2 or scores[0]<=0: return 1.0
    return (scores[0]-scores[1])/scores[0]

def cascade(lex_keys, lex_scores, neu_keys, gate, n_lex=3, n_neu=2):
    """If lexical confident -> lexical top5 unchanged. Else union lex[:n_lex]+neu[:n_neu]."""
    if lex_margin(lex_scores) >= gate:
        return lex_keys[:5]
    out=list(lex_keys[:n_lex])
    for k in neu_keys:
        if len(out)>=5: break
        if k not in out: out.append(k)
    # backfill from lexical if short
    for k in lex_keys:
        if len(out)>=5: break
        if k not in out: out.append(k)
    return out[:5]

def eval_order(order_fn, label):
    r1=r3=r5=0; n=0; reg=0; gain=0
    base_hit5_misses=[]
    for q,lk,ls,nk in data:
        gt,gtf=q['ground_truth'],q['gt_file']
        # baseline lexical hit@5
        base5 = find_rank(lk,gt,gtf,5) is not None
        order=order_fn(lk,ls,nk)
        rank=None
        for r,k in enumerate(order,1):
            if k==(gt,gtf): rank=r;break
        h5=rank is not None and rank<=5
        r1+=rank==1; r3+=(rank is not None and rank<=3); r5+=h5; n+=1
        if base5 and not h5: reg+=1
        if (not base5) and h5: gain+=1
    print(f'{label}: r@1={r1/n:.3f} r@3={r3/n:.3f} r@5={r5/n:.3f}  (recovered {gain}, regressed {reg})')
    return r5/n

print('\n=== CASCADE SWEEP (gate = lexical top1-top2 margin threshold) ===')
print('(gate=0 -> never consult neural = pure lexical; gate=1 -> always consult)')
base = eval_order(lambda lk,ls,nk: lk[:5], 'pure lexical (baseline)   ')
for gate in (0.05,0.10,0.15,0.20,0.30):
    eval_order(lambda lk,ls,nk,g=gate: cascade(lk,ls,nk,g), f'gated cascade @{gate:<4}      ')
