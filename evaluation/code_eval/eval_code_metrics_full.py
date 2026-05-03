#!/usr/bin/env python3
"""L1 code-structure evaluator for GISclaw agent runs.

Computes CodeBLEU, BLEU-4, ROUGE-L, edit similarity, and API operation F1
between agent-generated code and a gold reference. Paths are configured via
environment variables; see the README for the expected layout.
"""
import os, re, ast, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Per-cell agent run directories (override via env var)
DA_DIR = os.getenv("EVAL_DA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results", "dual_agent"))
SA_DIR = os.getenv("EVAL_SA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results", "single_agent"))
# Optional reference-code root (one Python file per task, name task-dependent)
TASK_DATA_ROOT = os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks"))
BENCH_DIR = TASK_DATA_ROOT
# Path to gold code; defaults to a per-task gold/ directory under SA_DIR
GOLD_CODE_DIR = os.getenv("EVAL_GOLD_CODE_DIR", os.path.join(PROJECT_ROOT, "results", "single_agent"))
TASKS = list(range(1, 51))
OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR",
                      os.path.join(PROJECT_ROOT, "evaluation", "eval_results"))
CELL_LABEL = os.getenv("EVAL_CELL_LABEL", "")  # e.g. "run1", "run2_SA"
EVAL_ARCHS = os.getenv("EVAL_ARCHS", "both")   # "SA" | "DA" | "both"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DA 
DA_MODELS = {
    "gpt-4.1":      {"path": "gpt-4.1",      "label": "GPT-4.1",      "color": "#2196F3"},
    "gpt-5.4":      {"path": "gpt-5.4",      "label": "GPT-5.4",      "color": "#1565C0"},
    "deepseek":     {"path": "deepseek-chat", "label": "DeepSeek",     "color": "#4CAF50"},
    "gemini-flash": {"path": "gemini-3-flash-preview", "label": "Gemini Flash", "color": "#9C27B0"},
    "llama-70b":    {"path": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "Llama-70B", "color": "#FF9800"},
    "qwen-14b":     {"path": "qwen2.5-coder-14b", "label": "Qwen-14B", "color": "#F44336"},
}
# SA 
SA_MODELS = {
    "gpt-4.1":      {"path": "gpt-4.1",      "label": "GPT-4.1",      "color": "#2196F3"},
    "gpt-5.4":      {"path": "gpt-5.4",      "label": "GPT-5.4",      "color": "#1565C0"},
    "deepseek":     {"path": "deepseek-chat", "label": "DeepSeek",     "color": "#4CAF50"},
    "gemini-flash": {"path": "gemini-3-flash-preview", "label": "Gemini Flash", "color": "#9C27B0"},
    "llama-70b":    {"path": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "Llama-70B", "color": "#FF9800"},
    "qwen-14b":     {"path": "qwen2.5-coder:14b", "label": "Qwen-14B", "color": "#F44336"},
}

# ============================================================
# ============================================================
def normalize_code(code):
    lines = []
    for line in code.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        lines.append(stripped)
    return '\n'.join(lines)

def compute_bleu4(reference, hypothesis):
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if len(hyp_tokens) == 0:
        return 0.0
    precisions = []
    for n in range(1, 5):
        ref_ng = Counter(ngrams(ref_tokens, n))
        hyp_ng = Counter(ngrams(hyp_tokens, n))
        if len(hyp_ng) == 0:
            precisions.append(0.0)
            continue
        clipped = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng)
        total = sum(hyp_ng.values())
        precisions.append(clipped / total if total > 0 else 0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(np.log(p) for p in precisions) / 4
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens)/len(hyp_tokens))
    return bp * np.exp(log_avg)

def compute_rouge_l(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref_tokens[i-1]==hyp_tokens[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n if n>0 else 0
    r = lcs/m if m>0 else 0
    return 2*p*r/(p+r) if (p+r)>0 else 0

def compute_edit_similarity(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    m, n = len(ref_tokens), len(hyp_tokens)
    if m==0 and n==0:
        return 1.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1] if ref_tokens[i-1]==hyp_tokens[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return 1.0 - dp[m][n]/max(m,n)

def compute_codebleu(reference, hypothesis):
    ref_norm = normalize_code(reference)
    hyp_norm = normalize_code(hypothesis)
    ngram_match = compute_bleu4(ref_norm, hyp_norm)
    # ngram
    python_kw = {'import','from','def','class','return','if','else','elif','for','while','try','except','with','as','in','not','and','or','True','False','None'}
    ref_t = ref_norm.split(); hyp_t = hyp_norm.split()
    ref_w = Counter(); hyp_w = Counter()
    for t in ref_t: ref_w[t] += (2.0 if t in python_kw else 1.0)
    for t in hyp_t: hyp_w[t] += (2.0 if t in python_kw else 1.0)
    weighted_ngram = sum(min(hyp_w[t], ref_w.get(t,0)) for t in hyp_w)/sum(hyp_w.values()) if sum(hyp_w.values())>0 else 0.0
    # AST
    def ast_subtrees(code, depth=3):
        try: tree = ast.parse(code)
        except SyntaxError: return Counter()
        sub = Counter()
        def walk(n,d):
            if d>depth: return type(n).__name__
            ch = [walk(c,d+1) for c in ast.iter_child_nodes(n)]
            sig = type(n).__name__
            if ch: sig += '('+','.join(sorted(ch))+')'
            sub[sig]+=1; return sig
        walk(tree,0); return sub
    rt = ast_subtrees(reference); ht = ast_subtrees(hypothesis)
    if rt and ht:
        cm = sum(min(ht[k],rt[k]) for k in ht if k in rt)
        p=cm/sum(ht.values()); r=cm/sum(rt.values())
        syntax_match = 2*p*r/(p+r) if (p+r)>0 else 0
    else: syntax_match = 0.0
    def dataflow(code):
        try: tree=ast.parse(code)
        except SyntaxError: return set()
        flows=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign):
                for t in n.targets:
                    if isinstance(t,ast.Name): flows.add(('DEF',t.id))
            elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load): flows.add(('USE',n.id))
            elif isinstance(n,ast.Call):
                if isinstance(n.func,ast.Attribute): flows.add(('CALL',n.func.attr))
                elif isinstance(n.func,ast.Name): flows.add(('CALL',n.func.id))
            elif isinstance(n,ast.Import):
                for a in n.names: flows.add(('IMPORT',a.name))
            elif isinstance(n,ast.ImportFrom) and n.module: flows.add(('IMPORT',n.module))
        return flows
    rf=dataflow(reference); hf=dataflow(hypothesis)
    if rf and hf:
        cm=len(rf&hf); p=cm/len(hf); r=cm/len(rf)
        dataflow_match=2*p*r/(p+r) if (p+r)>0 else 0
    else: dataflow_match=0.0
    codebleu = (ngram_match+weighted_ngram+syntax_match+dataflow_match)/4
    return {'codebleu':round(codebleu,4),'ngram_match':round(ngram_match,4),'weighted_ngram':round(weighted_ngram,4),'syntax_match':round(syntax_match,4),'dataflow_match':round(dataflow_match,4)}

def extract_api_calls(code):
    patterns = {
        r'gpd\.read_file|geopandas\.read_file':'read_vector', r'rasterio\.open':'read_raster',
        r'pd\.read_csv':'read_csv', r'\.buffer\(':'buffer', r'gpd\.sjoin|\.sjoin\(':'spatial_join',
        r'\.overlay\(|gpd\.overlay':'overlay', r'\.clip\(|gpd\.clip':'clip', r'\.dissolve\(':'dissolve',
        r'\.to_crs\(':'reproject', r'\.unary_union':'union', r'\.intersection\(':'intersection',
        r'\.convex_hull':'convex_hull', r'\.centroid':'centroid',
        r'rasterize|features\.rasterize':'rasterize', r'dst\.write|\.write\(':'raster_write',
        r'griddata|interpolate\.griddata':'interpolation', r'DBSCAN':'clustering_dbscan',
        r'KMeans':'clustering_kmeans', r'KernelDensity|gaussian_kde':'kde',
        r'plt\.savefig':'save_figure', r'\.plot\(':'plot', r'\.to_file\(':'save_vector',
        r'\.to_csv\(':'save_csv', r'groupby\(':'groupby', r'\.merge\(':'merge',
        r'zonal_stats|rasterstats':'zonal_stats', r'RandomForest':'random_forest',
    }
    found = set()
    for pat, name in patterns.items():
        if re.search(pat, code): found.add(name)
    return found

def compute_api_f1(ref_ops, hyp_ops):
    if not ref_ops and not hyp_ops: return {'precision':1.0,'recall':1.0,'f1':1.0}
    tp = len(ref_ops & hyp_ops)
    p = tp/len(hyp_ops) if hyp_ops else 0
    r = tp/len(ref_ops) if ref_ops else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    return {'precision':round(p,4),'recall':round(r,4),'f1':round(f1,4)}

# ============================================================
# ============================================================
def evaluate_arch(arch_name, results_dir, model_map):
    records = []
    total = 0
    for tid in TASKS:
        task_label = f"T{tid:02d}"
        # Primary gold location: results/<...>/T##/gold/code.py
        gold_path = os.path.join(GOLD_CODE_DIR, task_label, "gold", "code.py")
        if not os.path.exists(gold_path):
            # Fallback: a *.py file under <task_root>/<id>/
            gold_dir = os.path.join(BENCH_DIR, str(tid))
            py_files = [f for f in os.listdir(gold_dir) if f.endswith('.py')] if os.path.isdir(gold_dir) else []
            if py_files:
                gold_path = os.path.join(gold_dir, py_files[0])
            else:
                continue
        with open(gold_path, encoding='utf-8', errors='ignore') as f:
            gold_code = f.read()
        # gold ArcPy parse
        gold_norm = normalize_code(gold_code)
        gold_ops = extract_api_calls(gold_code)

        print(f"  [list] {task_label}: ", end="", flush=True)
        mc = 0
        for mk, mi in model_map.items():
            label = mi["label"]
            code_path = os.path.join(results_dir, task_label, mi["path"], "code.py")
            if not os.path.exists(code_path):
                records.append({'arch':arch_name,'task':task_label,'model':label,'exists':False,
                    'codebleu':0,'ngram_match':0,'weighted_ngram':0,'syntax_match':0,'dataflow_match':0,
                    'bleu4':0,'rouge_l':0,'edit_sim':0,'api_f1':0})
                continue
            with open(code_path, encoding='utf-8', errors='ignore') as f:
                model_code = f.read()
            model_norm = normalize_code(model_code)
            model_ops = extract_api_calls(model_code)
            cb = compute_codebleu(gold_code, model_code)
            bleu4 = compute_bleu4(gold_norm, model_norm)
            rouge_l = compute_rouge_l(gold_norm, model_norm)
            edit_sim = compute_edit_similarity(gold_norm, model_norm)
            api = compute_api_f1(gold_ops, model_ops)
            records.append({'arch':arch_name,'task':task_label,'model':label,'exists':True,
                'codebleu':cb['codebleu'],'ngram_match':cb['ngram_match'],'weighted_ngram':cb['weighted_ngram'],
                'syntax_match':cb['syntax_match'],'dataflow_match':cb['dataflow_match'],
                'bleu4':round(bleu4,4),'rouge_l':round(rouge_l,4),'edit_sim':round(edit_sim,4),'api_f1':api['f1']})
            total += 1; mc += 1
        print(f"[ok] ({mc})")
    print(f" {arch_name}: {total} ")
    return records

# ============================================================
# ============================================================
def visualize(df, output_path):
    models = ["GPT-4.1","GPT-5.4","DeepSeek","Gemini Flash","Llama-70B","Qwen-14B"]
    colors = {"GPT-4.1":"#2196F3","GPT-5.4":"#1565C0","DeepSeek":"#4CAF50","Gemini Flash":"#9C27B0","Llama-70B":"#FF9800","Qwen-14B":"#F44336"}
    metrics = ['codebleu','bleu4','rouge_l','edit_sim','api_f1']
    metric_labels = ['CodeBLEU','BLEU-4','ROUGE-L','Edit Sim','API F1']
    exist = df[df['exists']==True]

    fig = plt.figure(figsize=(26, 16))
    fig.suptitle("Code Quality Evaluation — DA Dual Agent + Single Agent (T01-T49)\n"
                 "6 Models | CodeBLEU + BLEU-4 + ROUGE-L + Edit Sim + API F1",
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30, top=0.91, bottom=0.06, left=0.06, right=0.97)

    # 1. DA vs SA (grouped bar, CodeBLEU)
    ax = fig.add_subplot(gs[0, 0])
    da = exist[exist['arch']=='DA'].groupby('model')['codebleu'].mean()
    sa = exist[exist['arch']=='SA'].groupby('model')['codebleu'].mean()
    mo = [m for m in models if m in da.index or m in sa.index]
    x = np.arange(len(mo)); w = 0.35
    bars1 = ax.bar(x-w/2, [da.get(m,0) for m in mo], w, label='Dual Agent', color='#6c8cff', alpha=0.85)
    bars2 = ax.bar(x+w/2, [sa.get(m,0) for m in mo], w, label='Single Agent', color='#ff8c6c', alpha=0.85)
    for b in bars1:
        if b.get_height()>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{b.get_height():.3f}", ha='center', fontsize=7)
    for b in bars2:
        if b.get_height()>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{b.get_height():.3f}", ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(mo, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, 0.45); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_title("CodeBLEU: DA vs Single Agent", fontsize=11, fontweight='bold')

    # 2. DA 
    ax = fig.add_subplot(gs[0, 1])
    da_df = exist[exist['arch']=='DA']
    mavg = da_df.groupby('model')[metrics].mean()
    mavg = mavg.reindex([m for m in models if m in mavg.index])
    x = np.arange(len(mavg)); w = 0.15
    for i, (met, lab) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x+i*w, mavg[met], w, label=lab, alpha=0.85)
    ax.set_xticks(x+w*2); ax.set_xticklabels(mavg.index, fontsize=7, rotation=25, ha='right')
    ax.set_ylim(0, 0.65); ax.legend(fontsize=6, loc='upper right'); ax.grid(axis='y', alpha=0.3)
    ax.set_title("DA Dual Agent — All Metrics", fontsize=11, fontweight='bold')

    # 3. CodeBLEU 
    ax = fig.add_subplot(gs[0, 2], polar=True)
    sub = ['ngram_match','weighted_ngram','syntax_match','dataflow_match']
    sub_lab = ['N-gram','Weighted','Syntax','Dataflow']
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); angles += angles[:1]
    da_df = exist[exist['arch']=='DA']
    for m in models:
        md = da_df[da_df['model']==m]
        if md.empty: continue
        vals = [md[s].mean() for s in sub]; vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=1.5, label=m, color=colors.get(m,'#888'), markersize=3)
        ax.fill(angles, vals, alpha=0.04, color=colors.get(m,'#888'))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(sub_lab, fontsize=8)
    ax.set_ylim(0, 0.5); ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.35,1.15))
    ax.set_title("DA CodeBLEU Sub-metrics", fontsize=11, fontweight='bold', pad=20)

    # 4. DA CodeBLEU heatmap
    ax = fig.add_subplot(gs[1, 0])
    da_df = exist[exist['arch']=='DA']
    pivot = da_df.pivot_table(index='model', columns='task', values='codebleu')
    pivot = pivot.reindex(index=[m for m in models if m in pivot.index])
    if not pivot.empty:
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.5)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("DA CodeBLEU by Task", fontsize=11, fontweight='bold')

    # 5. API F1 
    ax = fig.add_subplot(gs[1, 1])
    da_api = exist[exist['arch']=='DA'].groupby('model')['api_f1'].mean()
    sa_api = exist[exist['arch']=='SA'].groupby('model')['api_f1'].mean()
    mo2 = [m for m in models if m in da_api.index or m in sa_api.index]
    x = np.arange(len(mo2)); w = 0.35
    ax.bar(x-w/2, [da_api.get(m,0) for m in mo2], w, label='DA', color='#6c8cff', alpha=0.85)
    ax.bar(x+w/2, [sa_api.get(m,0) for m in mo2], w, label='SA', color='#ff8c6c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(mo2, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(0, 0.85); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_title("API F1: DA vs SA", fontsize=11, fontweight='bold')

    # 6. 
    ax = fig.add_subplot(gs[1, 2])
    da_comp = exist[exist['arch']=='DA'].groupby('model')[metrics].mean().mean(axis=1).sort_values(ascending=True)
    mc = [colors.get(m,'#888') for m in da_comp.index]
    bars = ax.barh(range(len(da_comp)), da_comp.values, color=mc, alpha=0.85)
    ax.set_yticks(range(len(da_comp))); ax.set_yticklabels(da_comp.index, fontsize=10)
    ax.set_xlim(0, 0.5)
    for b,v in zip(bars, da_comp.values):
        ax.text(v+0.005, b.get_y()+b.get_height()/2, f"{v:.3f}", va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_title("DA Composite Score (Equal Avg)", fontsize=11, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[ok] : {output_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("CodeBLEU (T01-T49 × 6 × 2 )")
    print("=" * 60)
    all_records = []
    if EVAL_ARCHS in ("DA", "both"):
        print(f"\n[package] DA Dual Agent: {DA_DIR}")
        all_records.extend(evaluate_arch("DA", DA_DIR, DA_MODELS))
    if EVAL_ARCHS in ("SA", "both"):
        print(f"\n[package] Single Agent: {SA_DIR}")
        all_records.extend(evaluate_arch("SA", SA_DIR, SA_MODELS))

    df = pd.DataFrame(all_records)
    suffix = f"_{CELL_LABEL}" if CELL_LABEL else ""
    csv_path = os.path.join(OUTPUT_DIR, f"codebleu_eval{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[stats] : {csv_path}")

    exist = df[df['exists']==True]
    print("\n" + "=" * 60)
    print("CodeBLEU ( × )")
    print("=" * 60)
    summary = exist.groupby(['arch','model'])[['codebleu','bleu4','rouge_l','edit_sim','api_f1']].mean()
    print(summary.round(3).to_string())

    img_path = os.path.join(OUTPUT_DIR, f"codebleu_eval{suffix}.png")
    visualize(df, img_path)
