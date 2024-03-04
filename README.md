# Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations

This is the implementation of the paper [Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations](https://arxiv.org/abs/2307.08678). 
## Table of Contents
* [Overview](#overview)
* [Code Structure](#code-structure)
* [How to Cite](#citation)

## Overview
Large language models (LLMs) are trained to imitate humans to explain human decisions. However, do LLMs explain themselves? Can they help humans build mental models of how LLMs process different inputs? To answer these questions, we propose to evaluate ***counterfactual simulatability*** of natural language explanations: whether an explanation can enable humans to precisely infer the model's outputs on diverse counterfactuals of the explained input. For example, if a model answers "yes" to the input question "Can eagles fly?" with the explanation "all birds can fly", then humans would infer from the explanation that it would also answer "yes" to the counterfactual input "Can penguins fly?". If the explanation is precise, then the model's answer should match humans' expectations.

We implemented two metrics based on counterfactual simulatability: *precision* and *generality*. We generated diverse counterfactuals automatically using LLMs. We then used these metrics to evaluate state-of-the-art LLMs (e.g., GPT-4) on two tasks: multi-hop factual reasoning and reward modeling. We found that LLM's explanations have **low precision** and that precision does not correlate with plausibility. Therefore, naively optimizing human approvals (e.g., RLHF) may not be a sufficient solution.

You could find more details of this work in our [paper](https://arxiv.org/abs/2307.08678).

## Code Structure
Code is within the `{strategyqa, shp}/` directory:

 - Metric Evaluation:
      - `task_qa` runs the model on the inputs to generate explanations and answers.

      - `simulate_qg.py` generates the counterfactuals conditioned on the model's explanation.

      - `simulate_qa.py` infers the model's answer on counterfactuals based on its explanations.

      - `calculate_generality.py` calculates the generality of the explanations as the diversity of simulatable counterfactuals. `utils/diversity_util.py` contains some helper functions for generality evaluation.

      - `calculate_precision.py` calculates the precision of the explanations as the fraction of counterfactuals where humans' expectations match the model's actual outputs.

      - We include the prompt and demonstration examples we use in `prompts/`.

 - Baselines:
      - `simulate_qg_polyjuice.py` generates counterfactuals using PolyJuice as opposed to LM prompting.

      - `generate_unnatural_explanations.py` forces the LM to generate explanations conditioned on the answer that it does not select.

## Questions?

If you have any questions related to the code or the paper, feel free to reach out to us at `yanda.chen@cs.columbia.edu`.

## Citation

```bibtex
@article{chen2023models,
  title={Do models explain themselves? counterfactual simulatability of natural language explanations},
  author={Chen, Yanda and Zhong, Ruiqi and Ri, Narutatsu and Zhao, Chen and He, He and Steinhardt, Jacob and Yu, Zhou and McKeown, Kathleen},
  journal={arXiv preprint arXiv:2307.08678},
  year={2023}
}
```