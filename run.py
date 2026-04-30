from src.case_study.pipeline import run_pipeline

output = run_pipeline()

print(f"documents: {output['result']['n_documents']}")
print(f"blocks: {output['result']['n_blocks']}")
print(f"queries: {output['result']['n_queries']}")
print(f"synthetic full queries: {output['result']['n_synthetic_full_queries']}")
print(f"best alpha: {output['alpha']:.2f}")
print(f"fallback threshold: {output['fallback_threshold']:.2f}")
print(output["overall"].round(4).to_string(index=False))
