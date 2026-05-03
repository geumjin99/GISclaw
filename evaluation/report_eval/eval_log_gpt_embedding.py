#!/usr/bin/env python3
"""
GPT Embedding text-embedding-3-large 
DA Dual Agent + Single Agent6 
Claude Gemini Pro
"""
import os, re, csv, warnings, time
import numpy as np
import pandas as pd
import openai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ============================================================
# Configuration (override via environment variables)
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DA_DIR = os.getenv("EVAL_DA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results", "dual_agent"))
SA_DIR = os.getenv("EVAL_SA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results", "single_agent"))
TASK_DATA_ROOT = os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks"))
BENCH_CSV = os.getenv("GISCLAW_TASK_INDEX", os.path.join(TASK_DATA_ROOT, "tasks.csv"))
BENCH_DIR = TASK_DATA_ROOT

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_EMBED_MODEL = "text-embedding-3-large"

TASKS = list(range(1, 51))
OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "evaluation", "eval_results"))
CELL_LABEL = os.getenv("EVAL_CELL_LABEL", "")
EVAL_ARCHS = os.getenv("EVAL_ARCHS", "both")  # "SA" | "DA" | "both"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6 Claude Gemini Pro
# DA 
DA_MODEL_MAP = {
    "gpt-4.1":      {"path": "gpt-4.1",      "label": "GPT-4.1",      "color": "#2196F3"},
    "gpt-5.4":      {"path": "gpt-5.4",      "label": "GPT-5.4",      "color": "#1565C0"},
    "deepseek":     {"path": "deepseek-chat", "label": "DeepSeek",     "color": "#4CAF50"},
    "gemini-flash": {"path": "gemini-3-flash-preview", "label": "Gemini Flash", "color": "#9C27B0"},
    "llama-70b":    {"path": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "Llama-70B", "color": "#FF9800"},
    "qwen-14b":     {"path": "qwen2.5-coder-14b", "label": "Qwen-14B", "color": "#F44336"},
}
# Single Agent qwen llama 
SA_MODEL_MAP = {
    "gpt-4.1":      {"path": "gpt-4.1",      "label": "GPT-4.1",      "color": "#2196F3"},
    "gpt-5.4":      {"path": "gpt-5.4",      "label": "GPT-5.4",      "color": "#1565C0"},
    "deepseek":     {"path": "deepseek-chat", "label": "DeepSeek",     "color": "#4CAF50"},
    "gemini-flash": {"path": "gemini-3-flash-preview", "label": "Gemini Flash", "color": "#9C27B0"},
    "llama-70b":    {"path": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "Llama-70B", "color": "#FF9800"},
    "qwen-14b":     {"path": "qwen2.5-coder:14b", "label": "Qwen-14B", "color": "#F44336"},
}

# ============================================================
# GPT Embedding API
# ============================================================
client = openai.OpenAI(api_key=OPENAI_API_KEY)
embed_cache = {}

def gpt_embed(text, max_tokens=8000):
    """ GPT text-embedding-3-large embedding"""
    text = text[:max_tokens]
    cache_key = hash(text)
    if cache_key in embed_cache:
        return embed_cache[cache_key]
    try:
        resp = client.embeddings.create(model=GPT_EMBED_MODEL, input=text)
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        embed_cache[cache_key] = emb
        return emb
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e):
            time.sleep(5)
            return gpt_embed(text, max_tokens)
        raise

def gpt_cosine(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    e1 = gpt_embed(text1)
    e2 = gpt_embed(text2)
    n1 = e1 / (np.linalg.norm(e1) + 1e-9)
    n2 = e2 / (np.linalg.norm(e2) + 1e-9)
    return float(np.dot(n1, n2))

# ============================================================
# execution_log
# ============================================================
def clean_execution_log(raw_text):
    lines = raw_text.split('\n')
    cleaned = []
    skip_traceback = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if 'OpenAI API initialized' in stripped or '[reason] Planner:' in stripped or '[tool] Worker:' in stripped:
            continue
        if 'Traceback (most recent call last)' in stripped:
            skip_traceback = True
            continue
        if skip_traceback:
            if stripped.startswith('Error') or stripped.startswith('Exception') or (': ' in stripped and not stripped.startswith(' ')):
                skip_traceback = False
                cleaned.append(f"Error: {stripped[:100]}")
            continue
        # Strip ASCII status tags emitted by the runner so embeddings see content only
        text = re.sub(r'\[(?:run|list|think|done|loop|ok|fail|warn|skip|write|stats|reason|tool|done|model|path|tip|trend|package|build|save|docs|alert|stop|guard|cost|wait|time|search)\]', '', stripped)
        text = re.sub(r'[═━-]{3,}', '', text)
        text = text.strip()
        if not text or re.match(r'^[\[\]\d\s:]+$', text):
            continue
        if len(text) > 200:
            text = text[:150] + "..."
        cleaned.append(text)
    return '\n'.join(cleaned)

# ============================================================
# Gold Reference 
# ============================================================
def load_gold_references():
    gold_refs = {}
    with open(BENCH_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row['id'])
            if tid not in range(1, 51):
                continue
            instruction = row.get('Instruction', '')
            workflow = row.get('Human Designed Workflow', '')
            gold_text = f"Task: {instruction}\n\nWorkflow: {workflow}"
            gold_code_dir = os.path.join(BENCH_DIR, str(tid))
            py_files = [f for f in os.listdir(gold_code_dir) if f.endswith('.py')] if os.path.isdir(gold_code_dir) else []
            if py_files:
                with open(os.path.join(gold_code_dir, py_files[0]), encoding='utf-8', errors='ignore') as cf:
                    code = cf.read()
                imports = [l for l in code.split('\n') if l.strip().startswith('import ') or l.strip().startswith('from ')]
                gold_text += f"\n\nKey libraries: {', '.join(imports[:10])}"
            sa_gold = os.path.join(SA_DIR, f"T{tid:02d}", "gold", "report.md")
            if os.path.exists(sa_gold):
                with open(sa_gold, encoding='utf-8') as rf:
                    gold_text = rf.read()
            gold_refs[tid] = gold_text
    return gold_refs

# ============================================================
# ============================================================
def evaluate_arch(arch_name, results_dir, model_map):
    """ GPT embedding """
    gold_refs = load_gold_references()
    records = []
    total = 0

    for tid in TASKS:
        task_label = f"T{tid:02d}"
        gold_text = gold_refs.get(tid, "")
        if not gold_text:
            continue

        print(f"  [list] {task_label}: ", end="", flush=True)
        model_count = 0

        for model_key, model_info in model_map.items():
            label = model_info["label"]
            log_path = os.path.join(results_dir, task_label, model_info["path"], "execution_log.txt")

            if not os.path.exists(log_path):
                records.append({"arch": arch_name, "task": task_label, "model": label,
                                "gpt_cosine": 0.0, "exists": False})
                continue

            with open(log_path, encoding='utf-8', errors='ignore') as f:
                raw_log = f.read()

            cleaned = clean_execution_log(raw_log)
            cos = gpt_cosine(gold_text, cleaned)
            records.append({"arch": arch_name, "task": task_label, "model": label,
                            "gpt_cosine": round(cos, 4), "exists": True})
            total += 1
            model_count += 1

        print(f"[ok] ({model_count})")

    print(f" {arch_name}: {total} ")
    return records

# ============================================================
# ============================================================
def visualize(df, output_path):
    models = ["GPT-4.1", "GPT-5.4", "DeepSeek", "Gemini Flash", "Llama-70B", "Qwen-14B"]
    colors = {"GPT-4.1": "#2196F3", "GPT-5.4": "#1565C0", "DeepSeek": "#4CAF50",
              "Gemini Flash": "#9C27B0", "Llama-70B": "#FF9800", "Qwen-14B": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Execution Log Semantic Evaluation — GPT text-embedding-3-large\n"
                 "Cleaned Log vs Gold Reference | T01-T49 × 6 Models",
                 fontsize=14, fontweight='bold', y=1.02)

    exist_df = df[df['exists'] == True]

    # 1. DA vs SA (grouped bar)
    ax = axes[0]
    da = exist_df[exist_df['arch'] == 'Dual Agent'].groupby('model')['gpt_cosine'].mean()
    sa = exist_df[exist_df['arch'] == 'Single Agent'].groupby('model')['gpt_cosine'].mean()
    model_order = [m for m in models if m in da.index or m in sa.index]
    x = np.arange(len(model_order))
    w = 0.35
    da_vals = [da.get(m, 0) for m in model_order]
    sa_vals = [sa.get(m, 0) for m in model_order]
    bars1 = ax.bar(x - w/2, da_vals, w, label='Dual Agent', color='#6c8cff', alpha=0.85)
    bars2 = ax.bar(x + w/2, sa_vals, w, label='Single Agent', color='#ff8c6c', alpha=0.85)
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.3f}",
                    ha='center', fontsize=8, fontweight='bold')
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.3f}",
                    ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("GPT Embedding Cosine Similarity")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title("DA Dual Agent vs Single Agent", fontsize=11, fontweight='bold')

    # 2. DA Heatmap (task × model)
    ax = axes[1]
    da_df = exist_df[exist_df['arch'] == 'Dual Agent']
    pivot = da_df.pivot_table(index='model', columns='task', values='gpt_cosine')
    pivot = pivot.reindex(index=[m for m in models if m in pivot.index])
    if not pivot.empty:
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=0.85)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("DA Dual Agent — Task × Model", fontsize=11, fontweight='bold')

    # 3. SA Heatmap
    ax = axes[2]
    sa_df = exist_df[exist_df['arch'] == 'Single Agent']
    pivot2 = sa_df.pivot_table(index='model', columns='task', values='gpt_cosine')
    pivot2 = pivot2.reindex(index=[m for m in models if m in pivot2.index])
    if not pivot2.empty:
        im2 = ax.imshow(pivot2.values, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=0.85)
        ax.set_xticks(range(len(pivot2.columns)))
        ax.set_xticklabels(pivot2.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(pivot2.index)))
        ax.set_yticklabels(pivot2.index, fontsize=9)
        plt.colorbar(im2, ax=ax, shrink=0.8)
    ax.set_title("Single Agent — Task × Model", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[ok] : {output_path}")

# ============================================================
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GPT Embedding (text-embedding-3-large)")
    print("6 × T01-T49 × 2 ")
    print("=" * 60)

    all_records = []

    if EVAL_ARCHS in ("DA", "both"):
        print(f"\n[package] DA Dual Agent: {DA_DIR}")
        all_records.extend(evaluate_arch("Dual Agent", DA_DIR, DA_MODEL_MAP))

    if EVAL_ARCHS in ("SA", "both"):
        print(f"\n[package] Single Agent: {SA_DIR}")
        all_records.extend(evaluate_arch("Single Agent", SA_DIR, SA_MODEL_MAP))

    df = pd.DataFrame(all_records)
    suffix = f"_{CELL_LABEL}" if CELL_LABEL else ""
    csv_path = os.path.join(OUTPUT_DIR, f"gpt_embedding_eval{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[stats] : {csv_path}")

    exist = df[df['exists'] == True]
    print("\n" + "=" * 60)
    print("GPT Embedding Cosine ( × )")
    print("=" * 60)
    summary = exist.groupby(['arch', 'model'])['gpt_cosine'].mean().unstack('arch')
    summary = summary.sort_values('Dual Agent', ascending=False)
    print(summary.to_string())

    print(f"\n[cost] API : {len(embed_cache)} unique embeddings")

    img_path = os.path.join(OUTPUT_DIR, "gpt_embedding_eval.png")
    visualize(df, img_path)
