# 🐾 Agentless-Java

**Agentless-Java** is an extension of the [Agentless](https://github.com/OpenAutoCoder/Agentless) framework, adapted to support automated fault localization for Java projects using Large Language Models (LLMs). This project focuses on replicating the Agentless pipeline in the Java ecosystem, addressing its object-oriented design, rigid type system, and other language-specific challenges.

> ⚠️ **Note**: This is a work-in-progress implementation. Our current focus is on the fault localization pipeline. Patch generation and validation are under development.

## 📁 Directory Overview

This folder contains the code and evaluation for fault localization on the SWE-bench-Java benchmark. It includes:

- File-level localization using both LLM prompting and embedding-based retrieval
- Element-level localization (classes/methods) via AST-based skeleton extraction
- Line-level fault localization with LLM + patch diff heuristics

For background, see our [📄 midterm report](agentless-java/Agentless-Java_MidtermReport.pdf).

---

## 🚀 How to Run the Pipeline

Ensure you are inside the `agentless-java/` directory and have installed all dependencies.

### 🔹 Step 1 & 2: Repository Structure + LLM-Based File Localization

Run:

```bash
python StructureTree.py
```

This generates a structured view of the Java repository and identifies suspicious files using Gemini-based LLMs.

> 🔑 **Important**: Replace the Gemini API key in the script with your own key.

---

### 🔹 Step 3: Embedding-Based File Localization

To perform embedding-based retrieval of suspicious files:

```bash
python agentlessstep3.py
```

Make sure to:
- Replace the Gemini API key
- Edit the path to the repository structure file (output from Step 1)

To evaluate file-level accuracy:

```bash
python agentlessstep3accueva.py
```

---

### 🔹 Step 4: Element-Level Localization

```bash
python batch_process.py \
  --input_file [input_data_file].json \
  --output_file [results_file].json \
  --llm [provider] \
  --api_key [your_api_key] \
  --model [model_name]
```

- The input file should be the suspicious file list from Step 3.
- Supported `--llm` values include: `openai`, `anthropic`, `gemini`.

---

### 🔹 Step 5: Line-Level Localization

```bash
python step5.py
```

- Input file: the JSON output from Step 4.
- Output: predicted line-level edit locations.

---

## 📊 Evaluation (from Midterm Report)

- **File-Level**
  - Superset Accuracy: 41.76%
  - Binary Touch Accuracy: 95.60%
- **Element-Level**
  - Binary Touch Accuracy: 90.00%
- **Line-Level**
  - Binary Touch Accuracy: 10.00%

These results demonstrate strong potential at early-stage localization and a solid foundation for downstream repair and validation.

---

## 📝 Citation

If you use this code or refer to our work, please cite our midterm report:

```bibtex
@misc{agentlessjava2025,
  title={Agentless-Java: Adapting Agentless Paradigm to Java Program Repair},
  author={Tianyi Huang and Wenqi Liao and Yiwei Wang and Yuyang Wang},
  year={2025},
  howpublished={CS 598 Midterm Report, University of Illinois Urbana-Champaign},
  url={https://github.com/upb3y/Agentless/tree/main/agentless-java}
}
```

---

## 🙌 Acknowledgements

This work builds on the original [Agentless](https://github.com/OpenAutoCoder/Agentless) framework and was evaluated using the [SWE-bench-Java](https://arxiv.org/abs/2408.14354) benchmark.

---
