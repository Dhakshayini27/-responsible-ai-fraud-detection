# -responsible-ai-fraud-detection
An MSc thesis project exploring fairness, explainability, and responsible AI in fraud detection using ANN, TabNet, SHAP, and a LangGraph multi-agent screening workflow.


# Responsible AI in Fraud Detection: Fairness, Explainability and Multi-Agent Screening

## Overview

This project investigates how fairness evaluation, explainable artificial intelligence (XAI), and multi-agent workflows can support more responsible AI-based fraud detection.

Developed as part of an MSc thesis in Human-Centred Artificial Intelligence at Technological University Dublin (TU Dublin), the project compares two machine learning models across two versions of a bank account fraud dataset. It examines age-related performance disparities, explores model explanations, and introduces a rule-based multi-agent workflow to screen explainability outputs for potentially concerning patterns that may require human review.

The project explores three research questions:

1. **Fairness:** How do ANN and TabNet compare in age-related fairness across the Base and Variant I datasets?
2. **Explainability:** How do SHAP explanations for ANN and native TabNet explanations represent feature importance across age groups?
3. **Responsible AI screening:** How can a multi-agent workflow use explainability outputs to identify potentially concerning patterns and support human oversight?

The aim is to demonstrate why responsible fraud detection requires more than predictive accuracy alone.

## Key Features

* Comparison of Artificial Neural Network (ANN) and TabNet models.
* Evaluation of age-related predictive disparities.
* Fairness metrics beyond accuracy.
* SHAP-based explanations for ANN predictions.
* Native feature-importance explanations from TabNet.
* Comparison of explanations across age groups and prediction outcomes.
* A LangGraph-based multi-agent workflow for rule-based explainability screening.
* Human-centred discussion of transparency, fairness, and responsible AI.

## Dataset

This project uses the **Bank Account Fraud (BAF) Dataset Suite**, introduced by Jesus et al. (2022).

The experiments use two dataset variants:

* **Base:** The reference dataset used for comparison.
* **Variant I:** An alternative dataset variant used to examine changes in representation, fraud prevalence, model behaviour, and fairness outcomes.

Each dataset contains approximately one million records and 32 original columns. The `month` column is excluded from the model inputs, leaving 30 input features.

For computational efficiency, a stratified sample of 100,000 records is selected from each dataset using `fraud_bool` as the stratification variable and `random_state=42`. Each sample is divided into training and testing sets using an 80:20 stratified split.

The Base dataset is treated as a **reference for comparison**, not as a guarantee of unbiased data or fair model behaviour.

**Dataset citation**

Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., and Bizarro, P. (2022). *BAF: A Benchmark for Modeling Bank Account Fraud*. NeurIPS 2022 Track on Datasets and Benchmarks.

* Paper: https://arxiv.org/abs/2208.04618
* Dataset repository: https://github.com/amazon-science/bank-account-fraud

Please consult the original dataset repository and its terms before redistributing dataset files. This repository can instead provide instructions for obtaining the data from its original source.

## Methodology

### 1. Data Preparation

The workflow includes:

* Loading and examining the Base and Variant I datasets.
* Sampling 100,000 records from each dataset using stratification by the fraud label.
* Excluding the `month` column from model inputs.
* Preparing model features and labels.
* Creating stratified 80% training and 20% testing splits.
* Comparing age-group representation and fraud prevalence.

Age is grouped into two categories:

* `<50`
* `>=50`

These groups are used for the selected fairness and explainability analyses.

### 2. Predictive Models

#### Artificial Neural Network (ANN)

The ANN is implemented using PyTorch. Its architecture consists of:

* Input layer.
* Fully connected hidden layer with 32 units and ReLU activation.
* Dropout with a rate of 0.3.
* Fully connected hidden layer with 16 units and ReLU activation.
* Dropout with a rate of 0.2.
* Output layer producing a fraud prediction logit.

The model uses `BCEWithLogitsLoss` with a positive-class weight and the Adam optimiser. Fraud probabilities are obtained by applying the sigmoid function to the logits, and a threshold of 0.5 is used for classification.

#### TabNet

TabNet is implemented using `pytorch-tabnet`. The configuration includes 32-dimensional decision and attention representations, five decision steps, a gamma value of 1.5, and the entmax feature-selection mask.

The model uses the Adam optimiser and a StepLR learning-rate scheduler, with early stopping during training. A threshold of 0.5 is used for classification.

Both models are evaluated on the Base and Variant I datasets using the same broad experimental approach.

### 3. Fairness Evaluation

The analysis evaluates age-related differences using the following metrics:

| Metric                        | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| True Positive Rate (TPR) Gap  | Difference in fraud detection rates among actual fraud cases |
| False Positive Rate (FPR) Gap | Difference in false alarm rates among non-fraud cases        |
| Equalized Odds Difference     | Summary of the largest selected TPR and FPR disparities      |
| Demographic Parity Difference | Difference in the proportion of records predicted as fraud   |
| Disparate Impact (DI) Ratio   | Ratio comparing positive prediction rates between groups     |

These metrics capture different aspects of model behaviour. No single metric provides a complete assessment of fairness, and their interpretation depends on the prediction task, group definitions, and operational context.

### 4. Explainability

#### SHAP for ANN

SHAP is used to investigate ANN predictions and the contribution of input features to individual predictions.

The analysis uses:

* A background sample of 100 training records selected with `random_state=42`.
* True positive (TP) and false positive (FP) prediction groups.
* Age-group comparisons between `<50` and `>=50`.
* Up to 50 records per subgroup.
* Mean absolute SHAP values to summarise feature importance.

The analysis examines whether protected or potentially sensitive attributes, such as customer age, appear prominently in model explanations.

#### Native TabNet Explanations

TabNet's native explanation outputs are used to examine feature importance across age groups for selected features, including:

* `customer_age`
* `employment_status`
* `income`

The results are compared between the Base and Variant I datasets.

Feature importance indicates how the model uses information for prediction. It does not, by itself, establish causality, prove discrimination, or determine whether a model is legally compliant.

### 5. Multi-Agent Explainability Screening

The project implements a rule-based multi-agent workflow using LangGraph.

The workflow contains three stages:

1. **Explanation Extractor:** Extracts relevant information from the explainability results.
2. **XAI Risk Screener:** Applies predefined rules to identify potentially concerning patterns in the prominence of protected or sensitive attributes.
3. **Report Generator:** Produces a structured summary of the screening results for human review.

The screening rules use feature-importance rankings and prominence windows. For example, a protected attribute appearing among the highest-ranked features can trigger a higher-priority review category.

The Top 3 and Top 5 windows used by the prototype are study-specific operational choices. They are not legal thresholds, validated measures of discrimination, or universal definitions of AI risk.

The workflow is intended to support transparency and human oversight. It does not independently calculate the full set of fairness metrics, establish discriminatory intent, or replace expert assessment.

## Key Findings

The experiments show that the observed age-related disparities differ by dataset variant and model.

The following table reports the selected fairness results as percentage-point gaps.

| Fairness Metric               | Base ANN | Variant I ANN | Base TabNet | Variant I TabNet |
| ----------------------------- | -------: | ------------: | ----------: | ---------------: |
| TPR Gap                       |   16.77% |         1.98% |      15.83% |            3.01% |
| FPR Gap                       |   10.95% |         2.91% |      20.96% |            8.32% |
| Equalized Odds Difference     |   16.77% |         2.91% |      20.96% |            8.32% |
| Demographic Parity Difference |   11.67% |         2.84% |      21.52% |            8.24% |

These results indicate smaller disparities across the selected metrics for Variant I than for Base, for both models in these experiments. This does not establish that Variant I is universally fairer or that either model is fair in an absolute sense.

The results also highlight that representation imbalance and predictive fairness are related but distinct. A dataset's group proportions alone do not determine the size or direction of model performance disparities.

### Explainability Findings

The explainability analysis shows that the prominence of age can vary by dataset variant, model, age group, and prediction outcome.

For ANN, customer age appears more prominently in the Base dataset's selected true-positive and false-positive explanations than in the corresponding Variant I summaries.

TabNet's native explanation results also show differences in age-related feature importance between the dataset variants and age groups.

These observations motivate further review of how models rely on sensitive or potentially sensitive attributes. They should not be interpreted as evidence that age caused a prediction or that a model necessarily discriminates against a group.

### Multi-Agent Screening Findings

The rule-based workflow converts selected explainability patterns into structured screening outputs that can help direct human attention.

The resulting flags are preliminary indicators for investigation. They are not independent proof of unfairness, legal non-compliance, or a requirement to reject or deploy a model.

## Responsible AI, MonTeReLo and Fundamental Rights Impact Assessment

This project is informed by the broader goal of making AI systems more transparent, accountable, and subject to meaningful human oversight.

### MonTeReLo

MonTeReLo is relevant to the wider discussion of monitoring and evaluating responsible AI systems. It provides a related reference point for considering how AI risks and system behaviour can be examined systematically.

The present project is narrower: its multi-agent prototype applies predefined rules to selected model-explanation outputs. It should not be presented as an implementation or validation of MonTeReLo unless a direct methodological relationship has been established.

**Reference:** Add the exact MonTeReLo publication, authors, year, and DOI or stable URL used in the thesis bibliography before publication. Confirm the spelling and bibliographic details against the original source.

### Fundamental Rights Impact Assessment (FRIA)

A Fundamental Rights Impact Assessment considers how an AI system may affect fundamental rights in its intended context of use. Depending on the applicable legal framework and system, this can require consideration of affected people, potential harms, safeguards, human oversight, and the circumstances in which the system is deployed.

The European Union's Artificial Intelligence Act includes FRIA obligations for certain deployers of high-risk AI systems under specified conditions. Whether a particular fraud detection system is subject to these obligations requires a contextual legal assessment.

This project's agent workflow is **not a complete FRIA tool**. It does not comprehensively assess fundamental rights, legal obligations, affected stakeholders, deployment context, or all relevant risks. Instead, it demonstrates one possible technical input into a broader human-led responsible AI review.

**Reference:**

European Union. (2024). *Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act).* Official Journal of the European Union.

* Official text: https://eur-lex.europa.eu/eli/reg/2024/1689/oj

## Technology Stack

* **Language:** Python
* **Data processing:** pandas, NumPy
* **Machine learning:** PyTorch, scikit-learn, pytorch-tabnet
* **Explainable AI:** SHAP, TabNet native explanations
* **Agent workflow:** LangGraph
* **Visualisation:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

Use the imports and versions in the final notebook to prepare an accurate `requirements.txt`.

## Repository Structure

An example structure for this repository is:

```text
responsible-ai-fraud-detection/
├── README.md
├── requirements.txt
├── dataset_part-4.ipynb
├── data/
│   ├── README.md
│   └── (dataset files, if redistribution is permitted)
└── images/
    ├── age_group_distribution.png
    ├── fraud_prevalence.png
    ├── fairness_comparison.png
    └── agent_workflow.png
```

Update the filenames to match the files actually committed to the repository. If the dataset cannot be redistributed, omit the CSV files and document how to obtain them from the original source.

## Limitations

* The experiments use a specific benchmark dataset and sampled subsets, so results may not generalise to real-world fraud systems.
* The analysis focuses on two age groups rather than a broad set of protected attributes or intersectional groups.
* Fairness results depend on the selected metrics, thresholds, group definitions, and experimental setup.
* Explainability outputs describe model behaviour but do not establish causation or prove discrimination.
* The agent workflow uses predefined, study-specific rules that require further validation.
* The prototype does not integrate all fairness metrics into its screening decisions.
* The workflow is not a complete FRIA, legal compliance assessment, or substitute for human judgement.

## Future Work

Potential extensions include:

* Evaluating the methods on additional and real-world datasets, where appropriate access and governance are available.
* Examining additional protected attributes and intersectional groups.
* Incorporating fairness metrics alongside explainability outputs in the screening workflow.
* Validating the screening rules with domain experts and affected stakeholders.
* Evaluating the workflow's reliability, false alarms, and usefulness for human reviewers.
* Extending the prototype to support more comprehensive human-led risk and fundamental-rights assessments.

## Reproducibility

For reproducibility:

1. Use the documented dataset source and preprocessing steps.
2. Install the required Python packages.
3. Run the notebook cells in order.
4. Keep the specified random seeds and train/test split settings consistent.
5. Verify that the reported metrics match the final notebook outputs.

Exact results can vary with software versions, hardware, and implementation details.

## Disclaimer

This repository is an academic research project. The models and agent workflow are experimental and are not intended for direct use in operational fraud decisions.

The outputs should not be treated as proof of discrimination, a legal opinion, a completed Fundamental Rights Impact Assessment, or a recommendation to automatically approve, reject, or freeze an account. Any real-world application requires appropriate validation, governance, legal review, and meaningful human oversight.

## Author

**Dhakshayini D. Janardhan**

MSc Human-Centred Artificial Intelligence
Technological University Dublin, Ireland

## References

1. Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., and Bizarro, P. (2022). *BAF: A Benchmark for Modeling Bank Account Fraud*. NeurIPS 2022 Track on Datasets and Benchmarks. https://arxiv.org/abs/2208.04618
2. European Union. (2024). *Regulation (EU) 2024/1689 (Artificial Intelligence Act).* https://eur-lex.europa.eu/eli/reg/2024/1689/oj
3. **MonTeReLo:** Insert the complete citation from the thesis bibliography, including author(s), year, title, publication or venue, and DOI or stable URL.
4. Lundberg, S. M., and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems. https://arxiv.org/abs/1705.07874
5. Arik, S. Ö., and Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. Proceedings of the AAAI Conference on Artificial Intelligence. https://arxiv.org/abs/1908.07442

---

*Before publishing, replace the MonTeReLo placeholder with the exact source cited in your thesis. Also verify the paper details, repository filenames, dependency versions, and dataset redistribution permissions.*
