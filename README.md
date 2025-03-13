# Vasospasm_Prediction
Severity Score Assignment by Artificial Intelligence Segmentation Tools for Prediction of Vasospasm in Subarachnoid Hemorrhage Patients

### Abstract
- Background and Motivation: This bachelor thesis explores the potential of artificial intelligence (AI) in risk stratification of patients with aneurysmal subarachnoid hemor- rhage (aSAH). aSAH is an acute cerebrovascular condition that can lead to severe com- plications such as vasospasm, which is presented as a focal neurological deficit. Despite the existence of various grading scales for assessing vasospasm risk, such as the modified Fisher scale, accurately predicting vasospasm remains a significant challenge. The perti- nence of this project was ensured by a systematic review conducted between September and December 2023.

- Objectives: The aim of this project is to automate the process of assigning the modi- fied Fisher scale to aSAH patients, and to integrate this scale with other baseline and CT features to enhance the prediction of vasospasm. Secondary objectives include the design of an accurate hemorrhage segmentation model (from which volume, density, thickness and localization measurements are computed) and the variability analysis between the modified Fisher Scale assignment of two neuroradiologists.

- Methods: Five different image segmentation techniques were evaluated. Two different classifiers were designed for the automated assessment of the modified Fisher Scale and the vasospasm prediction.
Results: The U-Net segmentation model achieved a Dice score of 95%, resulting in a precise differentiation of the hemorrhage and enabling accurate feature computation.
Regarding the modified Fisher classifier, the model was able to successfully assign higher risk of vasospasm - grade 4 (mean ROC AUC of 0.95 ± 0.08), moderate risk of vasospasm - grades 2 and 3 (mean ROC AUC of 0.80 ± 0.13) and lower risk of vasospasm - grades 0 and 1 (mean ROC AUC of 0.89 ± 0.10). It displayed a moderate Kappa coeffi- cient of 0.54 when compared to manual measurements.
The model for vasospasm prediction yielded a mean ROC AUC of 0.69 ± 0.16, out- performing the predictive capacity of current risk scales used in clinical practice. Input features for this model included hemorrhage volume and thickness, the modified Fisher scale and patient’s age.

- Conclusion: This project establishes the foundation for future approaches, fulfilling the stated objectives of the new AI-model for modified Fisher assignment and the ex- tended scale for vasospasm prediction. Further studies should explore more sophisticated AI methods to keep enhancing outcomes in this field.

- Keywords: Artificial intelligence (AI), aneurysmal subarachnoid hemorrhage (aSAH), vasospasm, delayed cerebral ischemia (DCI), modified Fisher scale, image segmentation, CT scan.
