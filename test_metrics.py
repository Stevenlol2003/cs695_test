import json
import csv
from pathlib import Path
from datetime import datetime
import statistics
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.io import load_theperspective_dataset, load_theperspective_evidence
from src.evaluation.local_metrics import recall_at_k, cover_at_k
from src.retrieval.tfidf_retrieval import retrieve_local_docs


def find_latest_results_file(results_dir: Path) -> Path:
    files = list(results_dir.glob('summary_results_*.json'))
    if not files:
        raise FileNotFoundError(f'No summary_results_*.json files in {results_dir}')
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def main(k: int = 5):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Find the latest results file produced by the pipeline
    results_file = find_latest_results_file(results_dir)
    print(f'Using results file: {results_file}')

    # Load dataset and evidence
    data = load_theperspective_dataset('data/theperspective')
    evidence = load_theperspective_evidence('data/theperspective')

    # Load pipeline results (contains queries)
    with open(results_file, 'r', encoding='utf-8') as f:
        pipeline_results = json.load(f)

    per_query_metrics = []
    recall_vals = []
    cover_vals = []

    for res in pipeline_results:
        query = res.get('query')
        if not query:
            continue

        # Retrieve top-k documents using TF-IDF on local evidence
        top_docs = retrieve_local_docs(query, evidence, k=k)
        retrieved_ids = [doc.get('id') for doc in top_docs]

        # Find gold ids for this query from the dataset
        gold_ids = []
        for doc in data:
            if doc.get('query') == query:
                gold_ids.extend(doc.get('favor_ids', []))
                gold_ids.extend(doc.get('against_ids', []))
                break

        if not gold_ids:
            print(f'Warning: no gold ids found for query: {query}')
            continue

        # Compute metrics
        r = recall_at_k(retrieved_ids, gold_ids, k=k)
        # `cover_at_k` does not accept a `k` argument (it computes coverage over sets)
        c = cover_at_k(retrieved_ids, gold_ids)

        per_query_metrics.append({
            'query': query,
            'recall_at_k': r,
            'cover_at_k': c,
            'retrieved_ids': retrieved_ids,
            'gold_ids': gold_ids,
        })

        recall_vals.append(r)
        cover_vals.append(c)

        print(f'Query: {query}')
        print(f'  recall@{k}: {r}')
        print(f'  cover@{k}:  {c}\n')

    # Aggregate
    mean_recall = statistics.mean(recall_vals) if recall_vals else 0.0
    mean_cover = statistics.mean(cover_vals) if cover_vals else 0.0

    print('Overall metrics:')
    print(f'  mean recall@{k}: {mean_recall:.4f}')
    print(f'  mean cover@{k}:  {mean_cover:.4f}')

    # Save per-query metrics to CSV with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = results_dir / f'metrics_{timestamp}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['query', 'recall_at_k', 'cover_at_k', 'retrieved_ids', 'gold_ids'])
        writer.writeheader()
        for row in per_query_metrics:
            writer.writerow({
                'query': row['query'],
                'recall_at_k': row['recall_at_k'],
                'cover_at_k': row['cover_at_k'],
                'retrieved_ids': json.dumps(row['retrieved_ids']),
                'gold_ids': json.dumps(row['gold_ids'])
            })

    print(f'Per-query metrics written to: {csv_file}')

    # Create a boxplot for per-query recall and cover
    try:
        recalls = [row['recall_at_k'] for row in per_query_metrics]
        covers = [row['cover_at_k'] for row in per_query_metrics]

        if recalls and covers:
            def quartiles(xs):
                xs = sorted(xs)
                n = len(xs)
                if n == 0:
                    return (math.nan, math.nan, math.nan)
                median = statistics.median(xs)
                q1 = xs[int(0.25 * (n - 1))]
                q3 = xs[int(0.75 * (n - 1))]
                return (q1, median, q3)

            recalls_q1, recalls_med, recalls_q3 = quartiles(recalls)
            covers_q1, covers_med, covers_q3 = quartiles(covers)
            recalls_mean = statistics.mean(recalls)
            covers_mean = statistics.mean(covers)
            # standard deviation (sample), fall back to 0.0 when insufficient data
            try:
                recalls_sd = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
            except Exception:
                recalls_sd = 0.0
            try:
                covers_sd = statistics.stdev(covers) if len(covers) > 1 else 0.0
            except Exception:
                covers_sd = 0.0

            plt.figure(figsize=(8, 6))
            colors = ['tab:blue', 'tab:orange']
            bplot = plt.boxplot([recalls, covers], labels=[f'recall@{k}', f'cover@{k}'], showmeans=True, patch_artist=True)
            # color boxes
            for i, box in enumerate(bplot['boxes']):
                box.set(facecolor=colors[i], alpha=0.6)
            for median in bplot['medians']:
                median.set(color='black')

            plt.ylabel('Score')
            plt.title(f'Boxplot of recall@{k} and cover@{k} (n={len(recalls)})')
            plt.grid(alpha=0.3)

            # Legend with key statistics
            recall_label = f"recall@{k}: mean={recalls_mean:.3f}, sd={recalls_sd:.3f}, median={recalls_med:.3f}, Q1={recalls_q1:.3f}, Q3={recalls_q3:.3f}"
            cover_label = f"cover@{k}:  mean={covers_mean:.3f}, sd={covers_sd:.3f}, median={covers_med:.3f}, Q1={covers_q1:.3f}, Q3={covers_q3:.3f}"
            handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], markersize=10),
                       Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=10)]
            plt.legend(handles, [recall_label, cover_label], loc='upper right')

            boxplot_file = results_dir / f'metrics_boxplot_{timestamp}.png'
            plt.tight_layout()
            plt.savefig(boxplot_file)
            plt.close()

            print(f'Boxplot saved to: {boxplot_file}')
        else:
            print('No per-query metrics to plot.')

    except Exception as e:
        print(f'Could not generate boxplot (matplotlib may be missing): {e}')


if __name__ == '__main__':
    main(k=5)