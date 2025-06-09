"""Evaluation script for the BRIGHT dataset with PyLate models"""

from __future__ import annotations

import argparse
import os

import mteb
import srsly

from pylate import evaluation, indexes, models, retrieve

if __name__ == "__main__":
    tasks = mteb.get_tasks(tasks=["BrightRetrieval"])
    tasks[0].load_data()
    parser = argparse.ArgumentParser(description="Query length")
    parser.add_argument(
        "--query_length",
        type=int,
        default="128",
    )
    args = parser.parse_args()
    query_length = args.query_length

    model_name = "lightonai/Reason-ModernColBERT"

    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
    )

    for eval_set in tasks[0].queries.keys():
        output_dir = f"BRIGHT_scores_{model_name.split('/')[-1]}_ir"
        # if file already exists, skip
        if os.path.exists(
            os.path.join(
                output_dir,
                f"{tasks[0].metadata.name}_{eval_set.replace('/', '_')}_evaluation_scores_qlen{query_length}.json",
            )
        ):
            print(
                f"Results already exist for {tasks[0].metadata.name} in {output_dir}. Continuing..."
            )
            continue

        index = indexes.PLAID(
            override=True,
            nbits=4,
            index_name=f"{tasks[0].metadata.name}_{eval_set}_{model_name.split('/')[-1]}_{query_length}_4bits_ir",
        )

        retriever = retrieve.ColBERT(index=index)

        documents_embeddings = model.encode(
            sentences=list(tasks[0].corpus[eval_set]["standard"].values()),
            batch_size=100,
            is_query=False,
            show_progress_bar=True,
        )

        index.add_documents(
            documents_ids=list(tasks[0].corpus[eval_set]["standard"].keys()),
            documents_embeddings=documents_embeddings,
        )
        queries_embeddings = model.encode(
            sentences=list(tasks[0].queries[eval_set]["standard"].values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=32,
        )

        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)
        filtered_scores = []
        # Excluding the "excluded_ids" from the scores
        for query_scores, excluded_ids in zip(
            scores, tasks[0].relevant_docs[eval_set]["excluded"].values()
        ):
            # Some splits have no excluded ids
            if excluded_ids == "N/A":
                filtered_scores.append(query_scores)
                continue
            filtered_query_scores = []
            for query_score in query_scores:
                if query_score["id"] in excluded_ids:
                    continue
                filtered_query_scores.append(query_score)
            filtered_scores.append(filtered_query_scores)

        evaluation_scores = evaluation.evaluate(
            scores=filtered_scores,
            qrels=tasks[0].relevant_docs[eval_set]["standard"],
            queries=list(tasks[0].queries[eval_set]["standard"].keys()),
            metrics=["map", "ndcg@1", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )

        print(evaluation_scores)
        output_dir = f"BRIGHT_scores_{model_name.split('/')[-2]}_ir"
        os.makedirs(output_dir, exist_ok=True)
        srsly.write_json(
            os.path.join(
                output_dir,
                f"{tasks[0].metadata.name}_{eval_set.replace('/', '_')}_evaluation_scores_qlen{query_length}.json",
            ),
            evaluation_scores,
        )