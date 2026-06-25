"""Workstream B eval: plain BM25+stem vs improved lexical retriever."""
import ast, os, re, json, sys, time
sys.path.insert(0, '/home/ubuntu/slmharness/src')
from openharness.orchestration.code_retriever import CodeRetriever, SymbolEntry
import tiktoken
from nltk.stem import PorterStemmer

ENC = tiktoken.get_encoding("cl100k_base")
PORTER = PorterStemmer().stem
REPOS = {'httpx':'/tmp/httpx','jinja2':'/tmp/jinja2','requests':'/tmp/requests',
         'flask':'/tmp/flask','click':'/tmp/click','rich':'/tmp/rich'}

def identity_split(tok): return [tok]

def extract_symbols(repo_root):
    syms = []
    for dp,_,files in os.walk(repo_root):
        for fn in files:
            if not fn.endswith('.py'): continue
            fp = os.path.join(dp, fn)
            try:
                tree = ast.parse(open(fp,errors='replace').read(), filename=fp)
            except SyntaxError: continue
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    doc = ast.get_docstring(node) or ""
                    syms.append(SymbolEntry(name=node.name,
                        file_path=os.path.relpath(fp,repo_root),
                        docstring=doc[:400], kind=type(node).__name__, line=node.lineno))
    return syms

def count_tokens(repo_root, rel_files):
    tot=0
    for rf in rel_files:
        try: tot += len(ENC.encode(open(os.path.join(repo_root,rf),errors='replace').read()))
        except Exception: pass
    return tot

# full-repo token totals
repo_total = {}
repo_syms = {}
for name,root in REPOS.items():
    syms = extract_symbols(root)
    repo_syms[name] = syms
    allpy = [os.path.relpath(os.path.join(d,f),root)
             for d,_,fs in os.walk(root) for f in fs if f.endswith('.py')]
    repo_total[name] = count_tokens(root, allpy)
    print(f'{name}: {len(syms)} symbols, full-repo {repo_total[name]} tokens', flush=True)

# build retrievers per repo
print('\nBuilding retrievers...', flush=True)
baseline = {}; improved = {}
for name,syms in repo_syms.items():
    baseline[name] = CodeRetriever(stemmer=PORTER, splitter=identity_split,
                                   test_penalty=1.0, class_boost=1.0).fit(syms)
    improved[name] = CodeRetriever(stemmer=PORTER, splitter=None,  # default=wordninja
                                   test_penalty=0.5, class_boost=1.0).fit(syms)
print('baseline splitter:', baseline['httpx'].splitter_name, '| improved splitter:', improved['httpx'].splitter_name)
print('stemmer:', baseline['httpx'].stemmer_name, flush=True)

# load queries
queries = [json.loads(l) for l in open('/home/ubuntu/slmharness/evals/context_retrieval_queries.jsonl')]
print(f'\n{len(queries)} queries loaded', flush=True)

def is_test_file(p):
    low=p.lower(); base=low.rsplit('/',1)[-1]
    return '/test' in low or low.startswith('test') or base.startswith('test_') or base.startswith('conftest')

def evaluate(retrievers, label, raw_path):
    agg = {'r1':0,'r3':0,'r5':0,'r10':0,'ctx':0,'red':0.0,'n':0}
    sub = {'impl':{'r1':0,'r3':0,'r5':0,'r10':0,'n':0}, 'test':{'r1':0,'r3':0,'r5':0,'r10':0,'n':0}}
    per_repo = {name:{'r5':0,'n':0} for name in REPOS}
    rawf = open(raw_path,'w')
    for q in queries:
        repo=q['repo']; gt=q['ground_truth']; gtf=q['gt_file']
        results = retrievers[repo].query(q['query'], top_k=10)
        rank=None
        for r in results:
            if r.entry.name==gt and r.entry.file_path==gtf:
                rank=r.rank; break
        top5_files=[]
        for r in results[:5]:
            if r.entry.file_path not in top5_files: top5_files.append(r.entry.file_path)
        ctx = count_tokens(REPOS[repo], top5_files)
        red = (repo_total[repo]-ctx)/repo_total[repo]*100 if repo_total[repo] else 0
        h1=rank is not None and rank<=1; h3=rank is not None and rank<=3
        h5=rank is not None and rank<=5; h10=rank is not None and rank<=10
        agg['r1']+=h1; agg['r3']+=h3; agg['r5']+=h5; agg['r10']+=h10
        agg['ctx']+=ctx; agg['red']+=red; agg['n']+=1
        per_repo[repo]['r5']+=h5; per_repo[repo]['n']+=1
        bucket='test' if is_test_file(gtf) else 'impl'
        sub[bucket]['r1']+=h1; sub[bucket]['r3']+=h3; sub[bucket]['r5']+=h5; sub[bucket]['r10']+=h10; sub[bucket]['n']+=1
        rawf.write(json.dumps({'repo':repo,'gt':gt,'gt_file':gtf,'query':q['query'],
            'rank':rank,'top5':[ {'name':r.entry.name,'file':r.entry.file_path,'score':round(r.score,3)} for r in results[:5]],
            'ctx_tokens':ctx})+'\n')
    rawf.close()
    n=agg['n']
    print(f'\n=== {label} (n={n}) ===')
    print(f"  recall@1={agg['r1']/n:.3f}  @3={agg['r3']/n:.3f}  @5={agg['r5']/n:.3f}  @10={agg['r10']/n:.3f}")
    print(f"  mean ctx tokens={agg['ctx']/n:.0f}  mean reduction={agg['red']/n:.1f}%")
    for b in ('impl','test'):
        s=sub[b]; m=s['n'] or 1
        print(f"  [{b} n={s['n']}] r@1={s['r1']/m:.3f} r@3={s['r3']/m:.3f} r@5={s['r5']/m:.3f} r@10={s['r10']/m:.3f}")
    print('  per-repo recall@5:')
    for name,d in per_repo.items():
        print(f"    {name}: {d['r5']}/{d['n']} = {d['r5']/(d['n'] or 1):.3f}")
    return {'label':label,'n':n,'r1':agg['r1']/n,'r3':agg['r3']/n,'r5':agg['r5']/n,'r10':agg['r10']/n,
            'ctx':agg['ctx']/n,'red':agg['red']/n,
            'impl_r5':sub['impl']['r5']/(sub['impl']['n'] or 1),'impl_n':sub['impl']['n'],
            'test_r5':sub['test']['r5']/(sub['test']['n'] or 1),'test_n':sub['test']['n'],
            'per_repo':{name:(d['r5'],d['n']) for name,d in per_repo.items()}}

t0=time.time()
res_base = evaluate(baseline,'BM25+stem baseline','/tmp/raw_bm25_baseline.jsonl')
res_impr = evaluate(improved,'Improved lexical (compound+defsite)','/tmp/raw_improved_lexical.jsonl')
print(f'\nElapsed {time.time()-t0:.1f}s')

print('\n=== LIFT (improved - baseline) ===')
for k in ('r1','r3','r5','r10'):
    print(f'  {k}: {res_base[k]:.3f} -> {res_impr[k]:.3f}  ({(res_impr[k]-res_base[k])*100:+.1f}pp)')
print(f"  impl r@5: {res_base['impl_r5']:.3f} -> {res_impr['impl_r5']:.3f}")
print(f"  test r@5: {res_base['test_r5']:.3f} -> {res_impr['test_r5']:.3f}")

json.dump({'baseline':res_base,'improved':res_impr,'repo_total':repo_total},
          open('/tmp/lexical_eval.json','w'), default=str)
print('\nSaved /tmp/lexical_eval.json')
