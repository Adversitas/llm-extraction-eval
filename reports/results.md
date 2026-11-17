# Results

## Extraction quality and cost

| model | strategy | precision | recall | micro_f1 | valid_first_pass | valid_final | hallucinated_per_100_docs | calls_per_doc | prompt_tokens_per_doc | completion_tokens_per_doc | ms_per_doc_p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct | zero-shot | 0.394 | 0.260 | 0.313 | 0.987 | 0.987 | 30.000 | 1.000 | 142.127 | 23.893 | 337.525 |
| qwen2.5-0.5b-instruct | zero-shot+repair | 0.394 | 0.260 | 0.313 | 0.987 | 0.993 | 30.000 | 1.013 | 145.960 | 24.027 | 337.525 |
| qwen2.5-0.5b-instruct | few-shot+repair | 0.221 | 0.492 | 0.305 | 0.990 | 1.000 | 45.000 | 1.010 | 301.020 | 38.077 | 639.714 |
| qwen2.5-0.5b-instruct | few-shot+repair+verify | 0.250 | 0.490 | 0.331 | 0.990 | 1.000 | 0.000 | 1.010 | 301.020 | 38.077 | 639.714 |
| qwen2.5-1.5b-instruct | zero-shot | 0.399 | 0.485 | 0.438 | 0.973 | 0.973 | 57.667 | 1.000 | 142.127 | 29.317 | 490.130 |
| qwen2.5-1.5b-instruct | zero-shot+repair | 0.399 | 0.485 | 0.438 | 0.973 | 0.973 | 57.667 | 1.027 | 150.023 | 31.860 | 490.130 |
| qwen2.5-1.5b-instruct | few-shot+repair | 0.427 | 0.542 | 0.478 | 0.977 | 0.997 | 19.667 | 1.023 | 304.747 | 29.310 | 515.737 |
| qwen2.5-1.5b-instruct | few-shot+repair+verify | 0.463 | 0.542 | 0.499 | 0.977 | 0.997 | 0.000 | 1.023 | 304.747 | 29.310 | 515.737 |
| qwen2.5-3b-instruct | zero-shot | 0.428 | 0.537 | 0.476 | 1.000 | 1.000 | 10.667 | 1.000 | 142.127 | 30.343 | 775.067 |
| qwen2.5-3b-instruct | zero-shot+repair | 0.428 | 0.537 | 0.476 | 1.000 | 1.000 | 10.667 | 1.000 | 142.127 | 30.343 | 775.067 |
| qwen2.5-3b-instruct | few-shot+repair | 0.490 | 0.710 | 0.580 | 1.000 | 1.000 | 13.667 | 1.000 | 296.127 | 29.997 | 624.456 |
| qwen2.5-3b-instruct | few-shot+repair+verify | 0.518 | 0.710 | 0.599 | 1.000 | 1.000 | 0.000 | 1.000 | 296.127 | 29.997 | 624.456 |

## F1 per entity type

| model | strategy | f1_persons | f1_organizations | f1_locations | f1_misc |
|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct | zero-shot | 0.098 | 0.314 | 0.489 | 0.000 |
| qwen2.5-0.5b-instruct | zero-shot+repair | 0.098 | 0.314 | 0.489 | 0.000 |
| qwen2.5-0.5b-instruct | few-shot+repair | 0.720 | 0.328 | 0.383 | 0.016 |
| qwen2.5-0.5b-instruct | few-shot+repair+verify | 0.730 | 0.349 | 0.442 | 0.018 |
| qwen2.5-1.5b-instruct | zero-shot | 0.663 | 0.393 | 0.378 | 0.000 |
| qwen2.5-1.5b-instruct | zero-shot+repair | 0.663 | 0.393 | 0.378 | 0.000 |
| qwen2.5-1.5b-instruct | few-shot+repair | 0.822 | 0.442 | 0.518 | 0.014 |
| qwen2.5-1.5b-instruct | few-shot+repair+verify | 0.829 | 0.461 | 0.562 | 0.015 |
| qwen2.5-3b-instruct | zero-shot | 0.716 | 0.472 | 0.523 | 0.060 |
| qwen2.5-3b-instruct | zero-shot+repair | 0.716 | 0.472 | 0.523 | 0.060 |
| qwen2.5-3b-instruct | few-shot+repair | 0.850 | 0.585 | 0.597 | 0.139 |
| qwen2.5-3b-instruct | few-shot+repair+verify | 0.867 | 0.604 | 0.628 | 0.142 |

## LLM-as-judge vs human annotation

| judge | judged | items | accuracy_vs_human | cohen_kappa | judge_yes_rate | TPR | TNR | sampled_run_to_run_agreement |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | qwen2.5-0.5b-instruct | 300 | 0.583 | 0.167 | 0.917 | 1.000 | 0.167 | 0.983 |
| qwen2.5-3b-instruct | qwen2.5-0.5b-instruct | 300 | 0.753 | 0.507 | 0.667 | 0.920 | 0.587 | 0.977 |

