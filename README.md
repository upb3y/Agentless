
# 🐾 Agentless-Java

**Agentless-Java** extends the original [Agentless](https://github.com/OpenAutoCoder/Agentless) framework to **fully automate Java program repair** with Large Language Models (LLMs).  
The pipeline now covers **fault localisation → patch generation → regression validation → patch ranking**, closing the loop from bug discovery to candidate fix selection.

---

## 📁 Directory Overview

| Folder / Script | Purpose |
|-----------------|---------|
| `StructureTree.py` | Build repository tree and prompt-based file-level localisation (LLM) |
| `agentlessstep3*.py` | Embedding retrieval, accuracy evals for file-level localisation |
| `batch_process.py` | Class/​method-level localisation via AST skeletons + LLM |
| `step5.py` | Line-level localisation with heuristic diff scoring |
| `repair/…` | LLM-based patch generation utilities (Step 6) |
| `gen_regr_test/run_test_for_dataset.py` | Dockerized **regression test harness** over a HF dataset (Step 7) |
| `gen_repro_test/rank_results.py` | Post-processing: rank & select passing patches (Step 8) |
| `docs/` | Reports, figures, and sample outputs |

*See our [📄 midterm report](agentless-java/Agentless-Java_MidtermReport.pdf) for the design rationale of Steps 1-5.*

---

## 🚀 End-to-End Pipeline

> Ensure you are inside `agentless-java/` and have installed all Python + Docker dependencies.  
> Wherever a script calls an LLM, **replace the API key placeholders** with your own credentials.

### 🔹 Step 1 & 2 — Repository Structure + LLM File Localization
```bash
python StructureTree.py          # builds AST & prompts Gemini (or OpenAI)
````

### 🔹 Step 3 — Embedding-Based File Localization

```bash
python agentlessstep3.py          # produces top-k suspicious files
python agentlessstep3accueva.py   # optional: compute accuracy against GT
```

### 🔹 Step 4 — Element-Level Localization

```bash
python batch_process.py \
  --input_file suspicious_files.json \
  --output_file element_preds.json \
  --llm anthropic \
  --api_key $CLAUDE_KEY \
  --model claude-3-sonnet-20240229
```

### 🔹 Step 5 — Line-Level Localization

```bash
python step5.py                   # emits line_preds.json
```

### 🔹 Step 6 — **Repair (Patch Generation)**

```bash
python repair/generate_patches.py \
  --bug_info line_preds.json \
  --llm openai \
  --api_key $OPENAI_KEY \
  --model gpt-4o-mini
```

The script streams multiple candidate patches per bug and stores them in
`generated_patches.jsonl`.

### 🔹 Step 7 — **Regression Validation**

```bash
python gen_regr_test/run_test_for_dataset.py \
  --dataset_name Daoguang/Multi-SWE-bench \
  --dataset_split java_verified \
  --output_file test_run_results.jsonl \
  --start_index 0 --end_index 50 \
  --timeout 1800 \
  --run_id sweep_$(date +%s)
```

Key flags (excerpt from the script):

| Flag              | Default                      | Description                                      |
| ----------------- | ---------------------------- | ------------------------------------------------ |
| `--dataset_name`  | `"Daoguang/Multi-SWE-bench"` | HF dataset to evaluate against                   |
| `--dataset_split` | `"java_verified"`            | Data split (`train`, `test`, etc.)               |
| `--timeout`       | `1800`                       | Seconds passed to the underlying Docker test run |

Each candidate patch is executed in an isolated Docker container running the project’s test suite. Results are appended to `test_run_results.jsonl`.

### 🔹 Step 8 — **Patch Ranking / Re-ranking**

```bash
python gen_repro_test/rank_results.py \
  --results_file test_run_results.jsonl \
  --patches_file generated_patches.jsonl \
  --output_file ranked_patches.jsonl
```

The ranking script promotes patches that (1) make previously failing tests pass,
(2) keep unrelated tests green, and (3) minimise the diff footprint.

---

## 📝 Citation

```bibtex
@misc{agentlessjava2025,
  title={Agentless-Java: From Fault Localisation to Fully Automated Repair in the Java Ecosystem},
  author={Tianyi Huang and Wenqi Liao and Yiwei Wang and Yuyang Wang},
  year={2025},
  howpublished={University of Illinois Urbana–Champaign, CS 598 Final Report},
  url={https://github.com/upb3y/Agentless/tree/main/agentless-java}
}
```

---

## 🙌 Acknowledgements

This project builds on the excellent [Agentless](https://github.com/OpenAutoCoder/Agentless) framework and evaluates fixes with the
[SWE-bench-Java](https://arxiv.org/abs/2408.14354) benchmark.

```
