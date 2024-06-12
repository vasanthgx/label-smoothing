

![logo](https://github.com/vasanthgx/label-smoothing/blob/main/images/logo.gif)


# Project Title


**Literature Review  - "When Does Label Smoothing Help?" published in 2019 by the Google Brain team**
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='right'>

<br><br>


## Introduction

- **Background**: It starts by acknowledging that neural network training is highly influenced by the loss function used for optimization.
- **Historical Context**: Initially, backpropagation was derived for minimizing quadratic loss, but researchers found that using cross-entropy loss often leads to better classification performance and faster convergence.
- **Exotic Objectives**: Even in the early days of neural network research, there were indications that using objectives other than cross-entropy could be beneficial.
- **Label Smoothing Introduction**: Szegedy et al. introduced label smoothing as a technique to improve accuracy. Instead of using "hard" targets (i.e., one-hot encoded targets), it uses a smoothed version of targets.
- **Applications**: Label smoothing has been successfully applied across various tasks such as image classification, speech recognition, and machine translation.

![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/table1.png)

### Contributions
1. **Visualization Method**: Introduces a novel visualization method based on linear projections of penultimate layer activations to understand differences between networks trained with and without label smoothing.
2. **Calibration of Predictions**: Label smoothing aligns the confidences of predictions with their accuracies.
3. **Effect on Distillation**: Shows that label smoothing negatively affects distillation, where student models perform worse when trained with label-smoothed teacher models due to loss of information in the logits.


![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/math1.png)

#### Mathematical Description of Label Smoothing:
Suppose we have a neural network that predicts probabilities for \( K \) classes. Let's denote:
- \( p_k \): Probability assigned to the \( k \)-th class by the model.
- \( w_k \): Weights and biases of the last layer.
- \( x \): Activations of the penultimate layer concatenated with a bias term.

For a neural network trained with hard targets, the standard cross-entropy loss is minimized. For label smoothing, we modify the targets before calculating the loss.

Let's see how label smoothing modifies the targets:

For a network trained with **hard targets**:
- True targets (\( y \)) are typically one-hot encoded vectors. For example:
  - If the true label for a sample is class 3 out of 5 classes: \( y = [0, 0, 1, 0, 0] \).

![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/math3.png)

### Examples:
Let's consider a simple example with 3 classes to illustrate:

#### Hard Targets:
If the true label is class 2:
- Hard targets: \( y = [0, 1, 0] \)

#### Label Smoothing:
Suppose we use a label smoothing parameter \( ϵ = 0.1 \):
- Modified targets:
  - For class 2 (true class): \( y_{LS} = [0.1, 0.9, 0.1] \) 
  - Other classes: \( y_{LS} = [0.05, 0.05, 0.9] \) (uniform distribution)

### Activations of Penultimate Layer:
- \( x \) represents the activations of the penultimate layer of the neural network.
- These activations are fed into the final layer to make predictions.
- Example: If \( x = [0.5, 0.8, 0.3] \) (activations for three neurons in the penultimate layer).

![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/math2.png)

## Penultimate Layer Representations

### Visualization Scheme

1. **Steps**:
   - Pick three classes.
   - Find an orthonormal basis of the plane crossing the templates of these three classes.
   - Project the activations of examples from these classes onto this plane.

2. **Results**:
   - This 2-D visualization shows how activations cluster around the templates and how label smoothing affects the distances between these clusters.

### Visualization Examples

1. **CIFAR-10 with AlexNet**:
   - Classes: "airplane", "automobile", "bird".
   - Without label smoothing: Clusters are broader and more spread out.
   - With label smoothing (factor of 0.1): Clusters are tighter and form regular triangles, indicating that examples are equidistant from all class templates.

2. **CIFAR-100 with ResNet-56**:
   - Classes: "beaver", "dolphin", "otter".
   - Similar behavior observed, with label smoothing leading to tighter clusters and better accuracy.
   - Without label smoothing: Higher absolute values in projections, indicating over-confident predictions.

3. **ImageNet with Inception-v4**:
   - Classes: "tench", "meerkat", "cleaver" (semantically different) and "toy poodle", "miniature poodle", "tench" (semantically similar).
   - With semantically similar classes:
     - Without label smoothing: Similar classes cluster close with isotropic spread.
     - With label smoothing: Similar classes form an arc, maintaining equidistant from all class templates.
   - Indicates that label smoothing helps in regularizing the distances even for fine-grained, semantically similar classes.
   
![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/visual1.png)

![alt text](https://github.com/vasanthgx/label-smoothing/blob/main/images/table2.png)
   
### Key Observations

- **Effect of Label Smoothing**:
  - It makes the activations of examples more structured by maintaining regular distances between different class templates.
  - It reduces over-confidence in predictions, as shown by the constrained difference in logits.
- **Independence from Architecture and Dataset**:
  - The impact of label smoothing is consistent across different architectures and datasets.
- **Erasure of Information**:
  - Label smoothing can erase fine-grained information, making classes more uniformly distant from each other, which might sometimes reduce the richness of the representation.





## Implicit model calibration

### Calibration in Neural Networks
- **Calibration** refers to how well the predicted probabilities of a model reflect the actual probabilities of the outcomes. A well-calibrated model's confidence scores match the actual accuracy.
- **Expected Calibration Error (ECE)** is a metric used to measure calibration. A lower ECE indicates better calibration.

### Modern Neural Networks and Calibration
- Guo et al. [15] demonstrated that modern neural networks often exhibit poor calibration despite high performance, tending to be overconfident.
- **Temperature Scaling** is a post-processing technique that improves calibration by scaling the logits (inputs to the softmax function) with a temperature parameter.

### Label Smoothing
- **Label Smoothing** is a technique that adjusts the hard labels by distributing some probability mass to all classes, thus preventing the network from becoming overconfident.
- The authors propose that label smoothing not only prevents overconfidence but also improves calibration, similar to temperature scaling.

### Image Classification Experiments
- The experiments involve training a ResNet-56 on CIFAR-100 and an unspecified network on ImageNet.
- **Reliability Diagrams** are used to visualize calibration. Perfect calibration is represented by a diagonal line where confidence equals accuracy.
- **Results**:
  - Without temperature scaling, models trained with hard targets are overconfident.
  - Temperature scaling improves calibration significantly.
  - Label smoothing also improves calibration, producing results comparable to temperature scaling.

### Machine Translation Experiments
- The experiments are conducted using the Transformer architecture on the English-to-German translation task.
- **BLEU Score** is the metric used to evaluate translation quality, while **Negative Log-Likelihood (NLL)** measures the likelihood of the correct sequence under the model.
- **Results**:
  - Label smoothing improves both BLEU score and calibration compared to hard targets.
  - Temperature scaling can improve calibration and BLEU score for hard targets but cannot match the BLEU score improvements achieved with label smoothing.
  - Label smoothing results in worse NLL, indicating a trade-off between calibration and likelihood.

### Key Findings
- Label smoothing effectively calibrates neural networks for both image classification and machine translation tasks.
- In image classification, label smoothing produces calibration similar to temperature scaling.
- In machine translation, label smoothing improves BLEU scores more than temperature scaling, even though it results in worse NLL.
- The relationship between calibration (ECE) and performance metrics (BLEU score) is complex, with label smoothing providing benefits that temperature scaling cannot fully replicate.




## Knowledge distillation
### Label Smoothing and Knowledge Distillation
- **Label Smoothing:** This technique improves the teacher network's accuracy by smoothing the labels, distributing some probability mass across all classes, which prevents the model from becoming overconfident.
- **Knowledge Distillation:** This process involves training a student network to mimic the teacher network. It uses a combination of the true labels and the teacher’s soft output (probabilities) to train the student.

### The Problem with Label Smoothing in Distillation
- When teachers are trained with label smoothing, although their accuracy improves, they produce inferior student networks compared to teachers trained with hard targets.
- Initial observations showed that a non-convolutional teacher trained on translated MNIST digits with hard targets and dropout achieved a low test error. Distilling this teacher to a student produced a reasonably accurate student. However, when the teacher was trained with label smoothing instead of dropout, despite faster training and slightly better performance, the resulting student performed worse.

### Mechanism of Distillation
- In distillation, the cross-entropy loss used for training is modified to include both the true labels and the soft outputs of the teacher.
- A parameter (ϵ) controls the balance between fitting the hard targets and approximating the softened teacher outputs.
- Temperature scaling is used to control the smoothness of the teacher's output, exaggerating differences between probabilities of incorrect answers.

### Experimental Setup and Findings
- Experiments were conducted using the CIFAR-10 dataset, with a ResNet-56 teacher and an AlexNet student.
- Four key results were analyzed:
  1. **Teacher’s Accuracy:** As a function of the label smoothing factor.
  2. **Student’s Baseline Accuracy:** As a function of the label smoothing factor without distillation.
  3. **Student’s Accuracy Post-Distillation:** With temperature scaling, using a teacher trained with hard targets.
  4. **Student’s Accuracy Post-Distillation:** Using a teacher trained with label smoothing.

### Smoothness Index
- To compare results, a smoothness index is defined. For scenarios involving label smoothing, it measures the mass allocated by the teacher to incorrect examples over the training set.
- Results showed that students distilled from teachers trained with hard targets outperformed those trained with label smoothing, indicating that the relative information between logits is lost when using label smoothing.

### Visualization and Information Erasure
- Visualizations of examples from the training set showed that hard targets resulted in broad clusters of examples, indicating varied similarities to other classes. Label smoothing, however, resulted in tight, equally separated clusters, indicating less variation in similarities.
- This "erasure" of information means that while label smoothing improves teacher accuracy, it hampers the distillation process because the nuanced information needed to distinguish between different classes is lost.

### Mutual Information
- The mutual information between the input and the logits was estimated to quantify information erasure.
- Results showed that while training, mutual information initially increased but then decreased, especially for networks trained with label smoothing. This confirmed that the collapse of representations into small clusters leads to the loss of distinguishing information, resulting in poorer student performance during distillation.

## Conclusion and Future work
- The authors conclude that while label smoothing enhances the teacher network's accuracy, **it negatively impacts the distillation process by erasing crucial information. Consequently, teachers trained with label smoothing are not necessarily better at transferring knowledge to student networks.**

**This detailed exploration highlights a trade-off when using label smoothing in the context of knowledge distillation: better teacher performance does not equate to better student performance due to the loss of informative nuances in the teacher's output**

- **Summary of Findings**: Many modern models use label smoothing, but its underlying inductive bias is not fully understood. The paper summarizes observed behaviors during training with label smoothing, focusing on how it encourages tight and equally distant clusters in penultimate layer representations, visualized with a new scheme.

- **Positive and Negative Effects**: Label smoothing improves generalization and calibration but can hinder distillation due to information erasure. It encourages treating incorrect classes equally probable, reducing structure in later representations and logit variation across predictions.

- **Future Research Direction**: The relationship between label smoothing and the information bottleneck principle is highlighted. Label smoothing reduces mutual information, suggesting a new research direction. Understanding this relationship could impact compression, generalization, and information transfer.

- **Implications for Calibration**: Extensive experiments show label smoothing's impact on implicit calibration of model predictions. This is crucial for interpretability and downstream tasks like beam-search that rely on calibrated likelihoods.

In essence, the paper emphasizes the need for further exploration into the relationship between label smoothing, information theory principles, and its implications for model compression, generalization, and calibration.



## References

1.	David E Rumelhart, Geoffrey E Hinton, Ronald J Williams, et al. Learning representations by back-propagating errors. [Nature.](https://www.nature.com/articles/323533a0)







## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

