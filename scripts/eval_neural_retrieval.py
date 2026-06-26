"""Workstream C: neural code-embedding retrieval (jina-embeddings-v2-base-code)."""
import ast, os, json, sys, time, subprocess
import numpy as np
sys.path.insert(0,'/home/ubuntu/slmharness/src')
import tiktoken
from sentence_transformers import SentenceTransformer

ENC=tiktoken.get_encoding("cl100k_base")
REPOS={'httpx':'/tmp/httpx','jinja2':'/tmp/jinja2','requests':'/tmp/requests',
       'flask':'/tmp/flask','click':'/tmp/click','rich':'/tmp/rich'}

def sig_of(node):
    try:
        a=[ar.arg for ar in node.args.args]
        return f"({', '.join(a)})"
    except Exception:
        return "()"

def extract(repo_root):
    syms=[]
    for dp,_,files in os.walk(repo_root):
        for fn in files:
            if not fn.endswith('.py'): continue
            fp=os.path.join(dp,fn)
            try: tree=ast.parse(open(fp,errors='replace').read(),filename=fp)
            except SyntaxError: continue
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    doc=ast.get_docstring(node) or ""
                    sig=sig_of(node) if not isinstance(node,ast.ClassDef) else ""
                    kind=type(node).__name__
                    text=f"{node.name}{sig}\n{doc[:500]}".strip()
                    syms.append({'name':node.name,'file':os.path.relpath(fp,repo_root),
                                 'kind':kind,'text':text})
    return syms

def count_tokens(root,files):
    tot=0
    for rf in files:
        try: tot+=len(ENC.encode(open(os.path.join(root,rf),errors='replace').read()))
        except Exception: pass
    return tot

print('Loading model...', flush=True)
MODEL='jinaai/jina-embeddings-v2-base-code'
m=SentenceTransformer(MODEL, trust_remote_code=True, device='cuda')
print('MODEL:',MODEL,'device:',m.device, flush=True)

repo_syms={}; repo_total={}; repo_emb={}
total_embedded=0; t_embed=0.0
for name,root in REPOS.items():
    syms=extract(root); repo_syms[name]=syms
    allpy=[os.path.relpath(os.path.join(d,f),root) for d,_,fs in os.walk(root) for f in fs if f.endswith('.py')]
    repo_total[name]=count_tokens(root,allpy)
    texts=[s['text'] for s in syms]
    t0=time.time()
    emb=m.encode(texts,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    dt=time.time()-t0; t_embed+=dt; total_embedded+=len(texts)
    repo_emb[name]=emb
    print(f'{name}: {len(syms)} symbols embedded in {dt:.1f}s ({len(texts)/dt:.0f}/s), shape {emb.shape}', flush=True)

vram=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
print(f'\nTotal embedded: {total_embedded} symbols in {t_embed:.1f}s = {total_embedded/t_embed:.0f} emb/s')
print(f'VRAM during inference: {vram}', flush=True)

queries=[json.loads(l) for l in open('/home/ubuntu/slmharness/evals/context_retrieval_queries.jsonl')]
# embed queries grouped by repo
from collections import defaultdict
byrepo=defaultdict(list)
for q in queries: byrepo[q['repo']].append(q)

def is_test(p):
    low=p.lower(); base=low.rsplit('/',1)[-1]
    return '/test' in low or low.startswith('test') or base.startswith('test_') or base.startswith('conftest')

agg={'r1':0,'r3':0,'r5':0,'r10':0,'ctx':0,'red':0.0,'n':0}
sub={'impl':{'r1':0,'r3':0,'r5':0,'r10':0,'n':0},'test':{'r1':0,'r3':0,'r5':0,'r10':0,'n':0}}
per_repo={name:{'r5':0,'n':0} for name in REPOS}
rawf=open('/tmp/raw_neural.jsonl','w')
t0=time.time()
for name,qs in byrepo.items():
    qtexts=[q['query'] for q in qs]
    qemb=m.encode(qtexts,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    sims=qemb @ repo_emb[name].T   # cosine (normalized)
    syms=repo_syms[name]
    for qi,q in enumerate(qs):
        order=np.argsort(-sims[qi])[:10]
        gt=q['ground_truth']; gtf=q['gt_file']
        rank=None
        for r,idx in enumerate(order,1):
            if syms[idx]['name']==gt and syms[idx]['file']==gtf:
                rank=r; break
        top5f=[]
        for idx in order[:5]:
            f=syms[idx]['file']
            if f not in top5f: top5f.append(f)
        ctx=count_tokens(REPOS[name],top5f)
        red=(repo_total[name]-ctx)/repo_total[name]*100 if repo_total[name] else 0
        h1=rank is not None and rank<=1;h3=rank is not None and rank<=3
        h5=rank is not None and rank<=5;h10=rank is not None and rank<=10
        agg['r1']+=h1;agg['r3']+=h3;agg['r5']+=h5;agg['r10']+=h10;agg['ctx']+=ctx;agg['red']+=red;agg['n']+=1
        per_repo[name]['r5']+=h5; per_repo[name]['n']+=1
        bk='test' if is_test(gtf) else 'impl'
        sub[bk]['r1']+=h1;sub[bk]['r3']+=h3;sub[bk]['r5']+=h5;sub[bk]['r10']+=h10;sub[bk]['n']+=1
        rawf.write(json.dumps({'repo':name,'gt':gt,'gt_file':gtf,'query':q['query'],'rank':rank,
            'top5':[{'name':syms[idx]['name'],'file':syms[idx]['file'],'sim':round(float(sims[qi][idx]),4)} for idx in order[:5]],
            'ctx_tokens':ctx})+'\n')
rawf.close()
n=agg['n']
print(f'\n=== NEURAL ({MODEL}) n={n} ===')
print(f"  recall@1={agg['r1']/n:.3f} @3={agg['r3']/n:.3f} @5={agg['r5']/n:.3f} @10={agg['r10']/n:.3f}")
print(f"  mean ctx tokens={agg['ctx']/n:.0f}  mean reduction={agg['red']/n:.1f}%")
for b in ('impl','test'):
    s=sub[b];mm=s['n'] or 1
    print(f"  [{b} n={s['n']}] r@1={s['r1']/mm:.3f} r@3={s['r3']/mm:.3f} r@5={s['r5']/mm:.3f} r@10={s['r10']/mm:.3f}")
print('  per-repo recall@5:')
for name,d in per_repo.items():
    print(f"    {name}: {d['r5']}/{d['n']} = {d['r5']/(d['n'] or 1):.3f}")
print(f'  query+rank elapsed {time.time()-t0:.1f}s')
json.dump({'model':MODEL,'n':n,'r1':agg['r1']/n,'r3':agg['r3']/n,'r5':agg['r5']/n,'r10':agg['r10']/n,
    'ctx':agg['ctx']/n,'red':agg['red']/n,'emb_per_s':total_embedded/t_embed,'vram':vram,
    'impl_r5':sub['impl']['r5']/(sub['impl']['n'] or 1),'test_r5':sub['test']['r5']/(sub['test']['n'] or 1),
    'impl_n':sub['impl']['n'],'test_n':sub['test']['n'],
    'per_repo':{k:(v['r5'],v['n']) for k,v in per_repo.items()}}, open('/tmp/neural_eval.json','w'))
print('Saved /tmp/neural_eval.json')
