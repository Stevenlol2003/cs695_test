import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from src.utils.io import load_theperspective_dataset
# from src.utils.io import load_theperspective_evidence
# from src.retrieval.tfidf_retrieval import retrieve_local_docs
# from src.retrieval.web_retrieval import search_web
# from src.validation.entailment import check_entailment
# from src.summarization.merge import merge_documents
# from src.summarization.merge import merge_docs_lists
from src.summarization.llm_summary import summarize_query
# from src.evaluation.web_metrics import evaluate_all


def main():
    # Command line arguments (examples):
    #   --dataset theperspective --offline-k 0 --online-k 10 --method tfidf --limit 10
    # Flags:
    #   --dataset   : theperspective | perspectrumx (perspectrumx not yet implemented)
    #   --offline-k : top-k offline (TF-IDF) docs (currently unused in pipeline)
    #   --online-k  : top-k web docs to retrieve via Tavily
    #   --method    : label baked into result filename (e.g., tfidf)
    #   --limit     : truncate dataset for quick tests
    parser = argparse.ArgumentParser(
        description="Web-Augmented Multi-Perspective Summarization Pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["theperspective", "perspectrumx"],
        required=True,
        default="theperspective",
        help="Dataset to use theperspective or perspectrumx"
    )
    parser.add_argument(
        "--offline-k",
        type=int,
        default=0,
        help="Number of top offline (TF-IDF) documents to retrieve."
    )
    parser.add_argument(
        "--online-k",
        type=int,
        default=5,
        help="Number of top online (web) documents to retrieve."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tfidf",
        help="Retrieval method label to include in filename (e.g., tfidf)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (e.g., 10 for a quick test)."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    offline_k = args.offline_k
    online_k = args.online_k
    method = args.method
    limit = args.limit

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"summary_results_{timestamp}.json"

    # Load dataset
    if dataset_name == "theperspective":
        dataset = load_theperspective_dataset("data/theperspective")
    else:
        raise NotImplementedError("Perspectrumx not yet added.")

    # Create claims dict
    claims_dict = {entry["query"]: entry["claims"] for entry in dataset}

    # total_queries = len(dataset)
    # print(f"\nLoaded {total_queries} queries from {dataset_name} dataset.")
    # print(f"Using top-{online_k} retrieval for web retrieval.")
    # print(f"Saving results to: {output_file}")

    # # Load evidence depending on dataset
    # if dataset_name == "theperspective":
    #     evidence = load_theperspective_evidence("data/theperspective")
    # else:
    #     raise NotImplementedError("Perspectrumx not yet added.")

    # # Load valid-web data for testing
    # valid_web_path = f"data/valid-web/valid-web-{online_k}.json"
    # with open(valid_web_path, 'r', encoding='utf-8') as f:
    #     valid_web_data = json.load(f)

    # Load merged data instead of full dataset
    merged_file = results_dir / f"merged-{online_k}.json"
    with open(merged_file, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)

    total_queries = len(merged_data)
    print(f"\nLoaded {total_queries} queries from {merged_file}.")
    print(f"Saving results to: {output_file}")

    # # web_docs_by_query = {item['query']: item['web_docs']['results'] for item in valid_web_data}

    # Optionally limit dataset for quick tests
    if limit is not None:
        merged_data = merged_data[:limit]
        print(f"Processing first {len(merged_data)} queries due to --limit={limit}.")

    # Go over each query
    results = []
    for i, entry in enumerate(merged_data):
        query_text = entry["query"]
        merged_corpus = entry["merged"]
        claims = claims_dict.get(query_text, [])
        print(f"[{i+1}/{len(merged_data)}] Summarizing: {query_text}")

        # # TF-IDF document retrieval
        # local_docs = retrieve_local_docs(query_text, evidence, k=offline_k)

        # # Web retrieval
        # web_docs = web_docs_by_query.get(query_text, [])

        # # Merge local documents + web documents
        # merged_corpus = merge_docs_lists(local_docs, web_docs)

        # Summarization
        summary = summarize_query(query_text, merged_corpus, claims)

        result_entry = {
            "query": query_text,
            "summary": summary
        }
        results.append(result_entry)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Summarization completed. Results saved to: {output_file}")

    # # Save all merged results
    # merged_file = results_dir / f"merged-{online_k}.json"
    # with open(merged_file, 'w', encoding='utf-8') as f:
    #     json.dump(merged_results, f, indent=2, ensure_ascii=False)

    # print(f"Pipeline completed")
    # print(f"Results saved to: {output_file}")
    # print(f"Merged docs saved to: {merged_file}")

if __name__ == "__main__":
    main()