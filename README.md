# Explainable AI for Cyber-Physical Systems

## Considerations on Implementation:

### XAI Criteria (from [1])
1. Comprehensibility/Interpretability: Capacity to express learned information in human-comprehensible manner.
- Natural language explanations (DoS because {features}...)
- Visual dashboard (feature heatmaps)
- Protocol specific terminology (tcp_syn_count: 'Connection attempts', etc)
2. Transparency: Comprehensible by itself without additional outside information.
- Self-explaining models
- Decision rule extraction (simplified rule tree from isolation tree)
- Feature importance intrinsic to model (eg: count how often a particular feature splits the branches)
3. Explainability: Interface between people and AI program.
- Different explanations for different users
- Counterfactuals (allow users to do a what if study; eg: what if packet rate was normal)
4. Understandability: Human can understand what model is doing without understanding inner workings.
- Analogies/metaphors (DoS: "like a traffic jam at intersection")
- Progressive disclosure (reveal more as users asks; first overview -> technical -> full details)
- Visual metaphors instead of technical plots (DoS: water tank filling)
5. Accuracy: Ability to correctly predict unseen instances.
- Comprehensive evaluation framework: offer evaluation on detection (typical ML metrics) as well as on explanation helpfulness (decision time, accuracy, compare user confidence/satisfaction)
- Cross-validation (k-fold) with explanations
6. Fidelity: Accurately represent what is going on inside black box model.
- Local fidelity measurement: how well explanation matches the black box; measure prediction correlation, feature importance correlation and decision boundary overlap
- Counterfactual fidelity: do explanations point to features that change the prediction; measure overlap of features changed from the minimal change vs features mentioned in explanation
- Global fidelity via distillation: train an interpretable model to mimic a black box with fidelity constraints (ie: 0.95 ~ close to black box)

### Key Performance Indicators (KPIs)

- Accuracy
'detection_accuracy': '>95%',
'false_positive_rate': '<1%',
'time_to_detection': '<100ms',

- Fidelity
'local_fidelity_score': '>0.9',
'counterfactual_consistency': '>0.8',

- Comprehensibility
'user_understanding_score': '>4/5',
'explanation_adoption_rate': '>80%',

- Transparency
'model_transparency_index': 'High',
'rule_coverage': '>90%',

- Practical
'soc_analyst_decision_time': 'Reduced by 50%',
'false_trip_prevention': '>95%'


### What Makes a 'Good' Explanation?
-  faithfulness: 0.85,  # Actually reflects model reasoning (see below for definition)
-  comprehensibility: "High",  # Understandable to security analysts
-  actionability: "Medium",  # Suggests specific next steps
-  conciseness: "Lists top 3 causes",  # Not overwhelming
-  confidence: "Provides uncertainty estimates",
-  consistency: "Similar anomalies get similar explanations"

### Evaluation Strategies for Faithfulness
Faithfulness: How well the explanation reflects what the model actually used to make its decision.
- Ablation: remove features the model says are important -> prediction should change.
- Counterfactual consistency: minimal changes to anomaly to return it to normal; what actually changed vs what does the model suggest changing?
- Local Fidelity (LIME-style): train a simple model (eg: LDA) and see if explanations differ with complex model; compare feature importance between them -> fidelity: how well those rankings correlate.
- Explanation Stability: similar inputs should yield similar explanations; get similar data, get the model to do the explanation, get all explanations and calculate stability by getting the variance of the explanations (they are feature vectors not sentences)


### References
[1] https://www.researchgate.net/publication/380285633_Explainable_AI_for_Cyber-Physical_Systems_Issues_and_Challenges