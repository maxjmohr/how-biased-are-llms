# Cognitive Biases in Large Language Models: An empirical analysis of state-of-the-art models

## Idea
This repository contains the code and collected data for my master thesis on cognitive biases in LLMs. For 8 cognitive biases, we recreate experiments with 4 different prompt scenarios (styles) and run them on 10 language models with each 5 different temperatures. Each experiment consists of 2 questions (control vs. test group) which are both prompted 100x each time. The model responses are converted into a *bias_detected* metric to allow for quantitative analysis and comparison between all experiment configurations.

## Structure

- **02_Thesis**: Contains the LaTeX files for the thesis
- **03_Codebase**: Contains the code for the experiments and detection analysis
- **04_Datasets**: Contains the raw model responses and the bias detection results