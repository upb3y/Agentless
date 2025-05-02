# 🐭 Agentless-Java

**Agentless-Java** extends the original [Agentless](https://github.com/OpenAutoCoder/Agentless) framework to **fully automate Java program repair** with Large Language Models (LLMs).  
The pipeline now covers **fault localization → patch generation → regression validation → patch ranking**, closing the loop from bug discovery to candidate fix selection.

---

## 📁 Directory Overview

| Folder / Script | Purpose |
|-----------------|---------|
| `StructureTree.py` | Build repository tree and prompt-based file-level localization (LLM) |
| `agentlessstep3*.py` | Embedding retrieval, accuracy evals for file-level localization |
| `batch_process.py` | Class/​method-level localization via AST skeletons + LLM |
| `step5.py` | Line-level localization with heuristic diff scoring |
| `repair/…` | LLM-based patch generation utilities (Step 6) |
| `gen_regr_test/...` | Dockerized **regression test harness** over a HF dataset (Step 7) |
| `gen_repro_test/...` | Generate reproduction test and rank & select passing patches (Step 8) |
| `docs/` | Reports, figures, and sample outputs |

*See our [📄 midterm report](agentless-java/Agentless-Java_MidtermReport.pdf) for the design rationale of Steps 1-5.*
*See our [📄 final report](agentless-java/Agentless-Java_FinalReport.pdf) for the whole pipeline.*


## 🚀 End-to-End Pipeline

> Ensure you are inside `agentless-java/` and have installed all Python + Docker dependencies.  
> Wherever a script calls an LLM, **replace the API key placeholders** with your own credentials.

### 🔹 Step 1 & 2 — Repository Structure + LLM File Localization
```bash
python StructureTree.py          # builds AST & prompts Gemini (or OpenAI)
````

### 🔹 Step 3 — Embedding-Based File Localization

```bash
agentlessstep3.py          # upload and run using colab: produces top-k suspicious files
agentlessstep3accueva.py   # upload and run using colab: optional: compute accuracy against GT
```
The output after step 3 will be `suspicious_files.json`.

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
python step5.py \
  --input_file suspicious_files.json

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

### 🔹 Step 8.1 — **Generate Reproduction Test Patches**

```bash
# Create the output directory first
mkdir ./generation_output

python generate_reproduction_tests.py \
    --output_folder ./generation_output \
    --filtered_patches_file ./filtered_patches_cleaned.jsonl \
    --max_samples 1 \
    --num_threads 4 \
    # Optional arguments:
    # --model <gemini_model_name>          # e.g., gemini-1.5-pro-latest
    # --dataset_name <huggingface_dataset> # For fetching problem statements
    # --dataset_split <split_name>
    # --target_id <specific_instance_id>   # For debugging one instance
```
This will create `output_0_processed_reproduction_test.jsonl` inside ./generation_output (assuming --max_samples 1).

### 🔹 Step 8.2 — **Run Reproduction Tests**

```bash
# Create the workspace directory first
mkdir ./repro_run_workspaces

python run_reproduction_tests.py \
    --repair_patch_file ./filtered_patches_cleaned.jsonl \
    --generated_test_file ./generation_output/output_0_processed_reproduction_test.jsonl \
    --workspace_dir ./repro_run_workspaces \
    --results_file ./test_run_results.jsonl \
    --num_workers 4 \
    --timeout 900 \
    # Optional arguments:
    # --cleanup_workspace             # Add flag to delete workspaces after runs
    # --dataset_name <hf_dataset>     # If not default
    # --dataset_split <split_name>    # If not default
    # --instance_ids <id1> <id2> ...  # To run only specific instances
    # --skip_existing                 # To resume a previous run
```
This generates `test_run_results.jsonl`, which is needed for the next step.

### 🔹 Step 8.3 — **Rank Patches**

```bash
# Create the workspace directory first
mkdir ./regr_rank_workspaces

python rank_results.py \
    --results_file ./test_run_results.jsonl \
    --patches_file ./filtered_patches_cleaned.jsonl \
    --original_results_dir ./original_passing_tests \
    --output_file ./ranked_reproduction_patches.jsonl \
    --workspace_dir ./regr_rank_workspaces \
    --num_workers 4 \
    --timeout 1800 \
    # Optional arguments:
    # --docker_image <image_name:tag>  # Override docker image for regression tests
```
The output file `ranked_reproduction_patches.jsonl` contains the final selected repair patch (including its content) for each instance where a valid, non-regressing, reproducing patch was found.



---

## 📝 Citation

```bibtex
@misc{agentlessjava2025,
  title={Agentless-Java: From Fault Localization to Fully Automated Repair in the Java Ecosystem},
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
