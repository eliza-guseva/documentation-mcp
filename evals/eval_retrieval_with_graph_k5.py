import ranx
from ranx import Qrels, Run
import pandas as pd 

eval_df = pd.read_csv('eval_df_with_graph_k5_n6.csv')

qrels_dict = {}
run_dict = {}
k_val = 5  # We are using k=5, quite arbitrary, but I noticed Claude tends to use k=5

# Prepare data for ranx using eval_df
for index, row in eval_df.iterrows():
    query_id = f"query_{index + 1}"  # Create a unique ID for each query (e.g., "query_1", "query_2")
    true_docs = row['true_5']
    returned_docs_tuple = row['returned_5']

    # Populate qrels: dictionary of relevant doc_ids with relevance score (1 for binary relevance)
    if true_docs is not None and isinstance(true_docs, list):
        qrels_dict[query_id] = {doc_id: 1 for doc_id in true_docs}
    else:
        qrels_dict[query_id] = {} # No relevant documents for this query or data malformed

    # Populate run: dictionary of retrieved doc_ids with a rank-based score
    # ranx expects higher scores to be better, so we invert the rank.
    if returned_docs_tuple is not None and isinstance(returned_docs_tuple, tuple):
        # Assign scores: e.g., for 5 docs, scores are 5 (rank 1), 4 (rank 2), ..., 1 (rank 5)
        run_dict[query_id] = {
            doc_id: len(returned_docs_tuple) - rank
            for rank, doc_id in enumerate(returned_docs_tuple)
        }
    else:
        run_dict[query_id] = {} # No retrieved documents for this query or data malformed


# Create Qrels and Run objects for ranx
qrels = Qrels(qrels_dict)
run = Run(run_dict)

# Define the metrics to calculate
# For F1-score, ranx uses 'f1_score@k'
metrics = [
    f"precision@{k_val}",
    f"recall@{k_val}",
    f"f1@{k_val}",
    f"ndcg@{k_val}",
    f"map@{k_val}",
    f"mrr@{k_val}"
]

# Calculate metrics
# ranx.evaluate returns a Report object
report = ranx.evaluate(qrels, run, metrics)

# Print the report
for (metric, value) in report.items():
    print(metric)
    print(value)
