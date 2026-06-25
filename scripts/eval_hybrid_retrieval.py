"""Workstream D: hybrid lexical+neural via Reciprocal Rank Fusion."""
import ast, os, json, sys, time
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
RRF_K=60; TOPN=50

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

def count_tokens(root,files):
    t=0
    for rf in files:
        try: t+=len(ENC.encode(open(os.path.join(root,rf),errors='replace').read()))
        except Exception: pass
    return t

print('Loading neural model...',flush=True)
m=SentenceTransformer('jinaai/jina-embeddings-v2-base-code',trust_remote_code=True,device='cuda')

repo_syms={};repo_total={};lex={};neu_emb={}
for name,root in REPOS.items():
    syms=extract(root); repo_syms[name]=syms
    allpy=[os.path.relpath(os.path.join(d,f),root) for d,_,fs in os.walk(root) for f in fs if f.endswith('.py')]
    repo_total[name]=count_tokens(root,allpy)
    ents=[SymbolEntry(name=s['name'],file_path=s['file'],docstring=s['doc'],kind=s['kind']) for s in syms]
    lex[name]=CodeRetriever(stemmer=PORTER,splitter=lambda t:[t],test_penalty=1.0).fit(ents)
    neu_emb[name]=m.encode([s['text'] for s in syms],batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    print(f'{name}: indexed {len(syms)}',flush=True)

queries=[json.loads(l) for l in open('/home/ubuntu/slmharness/evals/context_retrieval_queries.jsonl')]
byrepo=defaultdict(list)
for q in queries: byrepo[q['repo']].append(q)

def is_test(p):
    low=p.lower();base=low.rsplit('/',1)[-1]
    return '/test' in low or low.startswith('test') or base.startswith('test_') or base.startswith('conftest')

def rrf(lex_order, neu_order):
    score=defaultdict(float)
    for r,idx in enumerate(lex_order,1): score[idx]+=1.0/(RRF_K+r)
    for r,idx in enumerate(neu_order,1): score[idx]+=1.0/(RRF_K+r)
    return [i for i,_ in sorted(score.items(),key=lambda x:-x[1])]

agg={'r1':0,'r3':0,'r5':0,'r10':0,'ctx':0,'red':0.0,'n':0}
sub={'impl':{'r5':0,'n':0},'test':{'r5':0,'n':0}}
per_repo={name:{'r5':0,'n':0} for name in REPOS}
rawf=open('/tmp/raw_hybrid.jsonl','w')
for name,qs in byrepo.items():
    qemb=m.encode([q['query'] for q in qs],batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    sims=qemb @ neu_emb[name].T
    syms=repo_syms[name]
    for qi,q in enumerate(qs):
        lex_res=lex[name].query(q['query'],top_k=TOPN)
        # map lexical entries back to indices
        idx_by_key={(s['name'],s['file']):i for i,s in enumerate(syms)}
        lex_order=[idx_by_key.get((r.entry.name,r.entry.file_path)) for r in lex_res]
        lex_order=[i for i in lex_order if i is not None]
        neu_order=list(np.argsort(-sims[qi])[:TOPN])
        fused=rrf(lex_order,neu_order)[:10]
        gt=q['ground_truth'];gtf=q['gt_file']
        rank=None
        for r,idx in enumerate(fused,1):
            if syms[idx]['name']==gt and syms[idx]['file']==gtf: rank=r;break
        top5f=[]
        for idx in fused[:5]:
            f=syms[idx]['file']
            if f not in top5f: top5f.append(f)
        ctx=count_tokens(REPOS[name],top5f)
        red=(repo_total[name]-ctx)/repo_total[name]*100 if repo_total[name] else 0
        h1=rank is not None and rank<=1;h3=rank is not None and rank<=3
        h5=rank is not None and rank<=5;h10=rank is not None and rank<=10
        agg['r1']+=h1;agg['r3']+=h3;agg['r5']+=h5;agg['r10']+=h10;agg['ctx']+=ctx;agg['red']+=red;agg['n']+=1
        per_repo[name]['r5']+=h5;per_repo[name]['n']+=1
        bk='test' if is_test(gtf) else 'impl'; sub[bk]['r5']+=h5;sub[bk]['n']+=1
        rawf.write(json.dumps({'repo':name,'gt':gt,'gt_file':gtf,'rank':rank,'ctx_tokens':ctx})+'\n')
rawf.close()
n=agg['n']
print(f'\n=== HYBRID RRF (lexical BM25+stem + neural jina-code) n={n} ===')
print(f"  recall@1={agg['r1']/n:.3f} @3={agg['r3']/n:.3f} @5={agg['r5']/n:.3f} @10={agg['r10']/n:.3f}")
print(f"  mean ctx tokens={agg['ctx']/n:.0f}  reduction={agg['red']/n:.1f}%")
print(f"  [impl n={sub['impl']['n']}] r@5={sub['impl']['r5']/(sub['impl']['n'] or 1):.3f}")
print(f"  [test n={sub['test']['n']}] r@5={sub['test']['r5']/(sub['test']['n'] or 1):.3f}")
for name,d in per_repo.items(): print(f"    {name}: {d['r5']}/{d['n']} = {d['r5']/(d['n'] or 1):.3f}")
json.dump({'n':n,'r1':agg['r1']/n,'r3':agg['r3']/n,'r5':agg['r5']/n,'r10':agg['r10']/n,
    'ctx':agg['ctx']/n,'red':agg['red']/n}, open('/tmp/hybrid_eval.json','w'))
print('\n=== COMPARISON @5 ===')
print('  BM25+stem baseline: 0.966   (from Workstream B)')
print('  neural jina-code:   0.627   (from Workstream C)')
print(f"  hybrid RRF:         {agg['r5']/n:.3f}")
print('Saved /tmp/hybrid_eval.json')
