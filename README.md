# Machine Learning for Osteoporosis Detection using Annular Ring Feature Extraction

## Project Overview

This project details a comprehensive machine learning pipeline designed to classify bone X-ray images for osteoporosis detection. Recognizing the limitations of feeding raw image data into classical models, the core of this project is a custom feature engineering approach. By segmenting images into annular rings and extracting statistical and informational features, a high-signal, low-dimensional dataset was created. This structured dataset then served as the foundation for a systematic exploration of advanced tree-based ensemble models, culminating in a highly accurate and robust classifier.

## Tech Stack

- Data Processing & Analysis: Python, Pandas, NumPy, Scikit-learn, OpenCV
- Modeling: Scikit-learn (Decision Tree, Random Forest, Gradient Boosting), XGBoost, LightGBM
- Ensemble Methods: Voting Classifier, Stacking Classifier
- Model Evaluation & Interpretation: Matplotlib, Seaborn, SHAP, Calibration Plots

## 1. Feature Engineering: From Images to Insights

The primary challenge in medical image analysis is translating complex visual patterns into quantifiable features that a model can learn from. Instead of a "black-box" deep learning approach, this project focused on creating an interpretable feature set grounded in the pathophysiology of osteoporosis.

### 1.1. Annular Ring Segmentation

The chosen method was Annular Ring Segmentation. Each X-ray image was divided into 8 concentric rings originating from the center. This approach is powerful because it allows for a localized analysis of bone texture at different radial distances, mirroring how bone density can vary from the core to the periphery.

### 1.2. Rationale for Feature Selection

For each ring, four specific features were calculated. The choice of these features was deliberate, aiming to capture different aspects of bone health degradation:

- Mean Pixel Intensity: This is a direct proxy for Bone Mineral Density (BMD). In a grayscale X-ray, brighter areas (higher pixel values) correspond to denser material. A lower mean intensity in a ring suggests lower bone density, a primary indicator of osteoporosis.
- Standard Deviation: This measures the variance of pixel values within a ring, serving as a proxy for texture roughness and trabecular bone structure integrity. Healthy bone has a relatively uniform, albeit complex, internal structure. As osteoporosis progresses, this structure degrades, creating micro-fractures and increased porosity, which leads to higher pixel variance.
- Shannon Entropy: This quantifies the randomness or complexity of the pixel distribution. A uniform texture (e.g., all white or all black) has low entropy, while a highly complex, unpredictable texture has high entropy. Degraded bone tissue can present a more chaotic and less structured texture, making entropy a potentially powerful discriminative feature.
- Information Gain: This feature measures the reduction in entropy achieved by splitting the pixel values in a ring at their mean. It essentially quantifies the local contrast and separability of pixel intensities. High information gain suggests a bimodal distribution of light and dark patches within a ring, which could correspond to the porous, "pitted" appearance of osteoporotic bone.

This process converted each image into a 32-feature vector, which was compiled into a structured CSV file for modeling.

## 2. Exploratory Data Analysis (EDA)

Analysis of the generated dataset revealed key insights that validated the feature engineering approach and guided the modeling strategy.

- Class Balance: The dataset was confirmed to be well-balanced (approx. 2600 Normal vs. 3100 Osteoporosis samples), allowing for the use of accuracy as a supplementary metric, while prioritizing F1-score and Recall for a robust medical diagnosis evaluation.
- Feature Trends: The plot of mean pixel values across the 8 rings provided strong validation for the methodology. It showed a clear and consistent separation between the two classes, with the "Normal" class exhibiting higher mean values (denser bone) across almost all rings. This confirmed that the engineered features successfully captured clinically relevant information.
- Criticality of Entropy: The boxplots for entropy features revealed what initially appeared to be numerous outliers. However, a crucial experiment was conducted where these "outliers" were capped. This resulted in a significant 17% drop in model accuracy. This finding was pivotal, proving that these extreme values were not noise but were in fact high-signal indicators of significant textural differences, and were essential for the model's predictive power.

## 3. Modeling Strategy: A Focus on Tree-Based Ensembles

The modeling phase began with a baseline Decision Tree classifier. Its surprisingly strong performance on the structured feature set indicated that the data was well-suited for tree-based logic (i.e., decision boundaries based on feature thresholds). This insight motivated a focused and systematic exploration of more advanced tree-based ensemble techniques, which are state-of-the-art for tabular data.

The models were grouped into two categories:

1. Bagging Models: Random Forest and Extra Trees, which reduce variance by averaging predictions from multiple trees trained on different data subsets.
2. Boosting Models: Gradient Boosting, XGBoost, and LightGBM, which build trees sequentially, with each new tree correcting the errors of the previous ones.

## 4. Final Model Selection and Performance Analysis

After a rigorous evaluation of all individual ensemble models using 5-fold cross-validation, it was clear that a combination of these strong learners could yield the best result. A VotingClassifier was selected as the final model, as it demonstrated the highest and most stable performance.

### 4.1. The Voting Ensemble & The Importance of Diversity

The final model is a "soft voting" ensemble that combines the predictions of the four top-performing tree-based models:

- LightGBM Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Random Forest Classifier

By averaging the predicted probabilities from each of these diverse models, the VotingClassifier leverages their individual strengths and mitigates their weaknesses, leading to a more robust and accurate final prediction than any single model could achieve alone.

An important finding was made during this process: while hyperparameter tuning each base model individually is a standard technique, applying it here resulted in a slight decrease in the ensemble's overall performance (from 0.8533 to 0.8527 Mean CV F1-Score). This counter-intuitive result highlights a key principle of ensemble modeling: diversity is as important as individual model strength. Hyper-tuning each model made them individually stronger but also more similar in their decision-making, reducing the "second opinion" benefit of the ensemble. Therefore, the final model uses the robust default parameters for its base estimators to maintain this crucial model diversity. This approach achieved a Mean Cross-Validated F1-Score of 0.8533.

### 4.2. Final Performance on Unseen Data

The trained ensemble was evaluated on a held-out test set to simulate real-world performance.

| Metric                   | Score | Interpretation                                                                 |
|--------------------------|-------|---------------------------------------------------------------------------------|
| Osteoporosis Recall      | 0.91  | Primary Success Metric. The model correctly identifies 91% of patients who actually have osteoporosis. |
| Osteoporosis F1-Score    | 0.86  | Indicates an excellent balance between precision and recall for the positive class. |
| Osteoporosis Precision   | 0.82  | When the model predicts osteoporosis, it is correct 82% of the time.          |
| Overall Accuracy         | 0.84  | The model's predictions are correct 84% of the time across both classes.       |
| Macro Average F1-Score   | 0.84  | Shows strong, balanced performance across both Normal and Osteoporotic classes. |

The high recall score is the most significant result, as the primary goal of a medical screening tool is to minimize false negatives and correctly identify those who need further attention.

### 4.3. Model Validation and Interpretation

- Model Calibration: The calibration curve for the VotingClassifier was plotted and found to be very close to the ideal diagonal line. This demonstrates that the model's predicted probabilities are reliable and can be trusted as true confidence scores.

- Explainable AI (XAI) with SHAP: To look inside the model's "black box," the SHAP (SHapley Additive exPlanations) library was used. SHAP assigns each feature an "importance" value for every single prediction, allowing us to understand the model's logic at a granular level. The analysis revealed a clear narrative:

  - Feature Impact: The SHAP summary plots confirmed that the model's predictions are most heavily influenced by the texture of the innermost (ring 0) and outermost (rings 6 & 7) regions of the bone.
  - Prediction Logic: The beeswarm plot provided deeper insight. Each point on the plot represents a prediction for a single X-ray. Red points indicate high feature values, and blue points indicate low feature values.
    - std_0 (rougher core texture) and entropy_6 (more random edge texture) showed a clear trend: high values (red points) had positive SHAP values, strongly pushing the model's output higher and leading to a prediction of Osteoporotic.
    - Conversely, low values for these features (blue points) had negative SHAP values, pushing the prediction towards Normal. This analysis provides a data-driven confirmation of the medical characteristics of the disease and proves that the feature engineering approach successfully captured interpretable and powerful signals from the raw images.

## 5. Conclusion

This project successfully demonstrates the power of combining domain-specific feature engineering with a systematic exploration of advanced machine learning models. The final ensemble classifier is not only accurate but also robust and interpre
