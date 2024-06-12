

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

For a network trained with **label smoothing**:
- Modified targets (\( y_{LS} \)) are a mixture of hard targets and a uniform distribution.
- \( y_{LS_k} = y_k(1 - \epsilon) + \epsilon / K \), where \( \epsilon \) is the smoothing parameter and \( K \) is the number of classes.

#### Mathematical Notations:
- **Cross-Entropy Loss \( H(y; p) \)**: It measures the dissimilarity between the true distribution (\( y \)) and the predicted distribution (\( p \)).
  - For hard targets (\( y \)): \( H(y; p) = -\sum_{k=1}^{K} y_k \log(p_k) \)
  - For label smoothing (\( y_{LS} \)): \( H(y_{LS}; p) = -\sum_{k=1}^{K} y_{LS_k} \log(p_k) \)

### Examples:
Let's consider a simple example with 3 classes to illustrate:

#### Hard Targets:
If the true label is class 2:
- Hard targets: \( y = [0, 1, 0] \)

#### Label Smoothing:
Suppose we use a label smoothing parameter \( \epsilon = 0.1 \):
- Modified targets:
  - For class 2 (true class): \( y_{LS} = [0.1, 0.9, 0.1] \) 
  - Other classes: \( y_{LS} = [0.05, 0.05, 0.9] \) (uniform distribution)

### Activations of Penultimate Layer:
- \( x \) represents the activations of the penultimate layer of the neural network.
- These activations are fed into the final layer to make predictions.
- Example: If \( x = [0.5, 0.8, 0.3] \) (activations for three neurons in the penultimate layer).


## Previous work

Fine-grained recognition is a topic of large practical importance and many recent works have addressed such tasks including recognition of flowers , birds , cats and dogs, tree-leaves.
Segmentation has played an important role in object recognition with many algorithms available
In another body of works, called co-segmentation, better models are trained by exploiting shared appearance features in images containing the same class of objects.
These approaches are either too slow or are targeted for segmentation during training.
Segmentation has been popular as an initial step for object detection  or scene interpretation.
Those methods typically work with small coherent regions on the image and feed the low-level segmentations to object detection pipelines.
Other related work, not doing segmentation per se, has proposed to first localize the potential object region and utilize this information during recognition.

## Object detection and segmentation

This section describes how to detect and segment the object, or objects, in an image
A set of rudimentary region-based detection of parts of the object are done.
Using those regions as initialization, the Laplacian propagation method, presented, is applied.
The segmented image and input image are processed through the feature extraction and classification pipeline and the final classification is obtained

### Understanding the Laplacian Propagation in Semi-Supervised Learning

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp1.png)

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp2.png)

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp3.png)

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp4.png)




## Detecting object-specific regions
The authors start the method with an initial search for regions possibly belonging to an object from the super-class.
Using the features above, the authors train a classification model to decide if a region belongs to a super-class or the background.
Using ground truth segmentation of training images, the authors consider super-pixel regions with large overlap with the foreground and background ground truth areas, as positive and negative examples, respectively.
When no ground truth is available, the authors start from an approximate segmentation and iteratively improve the segmentation by applying the trained model.
Each model is used to segment the training images anew; the newly segmented images are used as ‘ground truth’ for building an improved model, and so on.
This procedure is standard in other segmentation works.
As shown later in the experiments, the authors have the same algorithms for both training of the model and detection for flowers, birds, cats and dogs

## Full-object segmentation
Let Ij denote the j-th pixel in an image and fj denotes its feature representation. The goal of the segmentation task is to find the label Xj for each pixel Ij, where Xj = 1 when the pixel belongs to the object and Xj = 0, otherwise.
The authors set fi to be the (R,G,B) color values of the pixel, mostly motivated by speed of computation, but other choices are possible too.
Djj i=1 where Dii = j=1 N W ij and Y are the desired labels for some the pixels
Those label constraints can be very useful to impose prior knowledge of what is an object and background.
This is a standard Laplacian label propagation formulation, and the equation above is often written in an equivalent and more convenient form: C(X) = XT (I − S)X + λ|X − Y |2.

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp7.png)

## Optimization
The optimization problem can be solved iteratively. Alternatively, it can be solved as a linear system of equations, which is the approach the authors chose.
After differentiation  the authors obtain an optimal solution for X, which the authors solve as a system of linear equations: In the implementation the authors use the Conjugate Gradient method, with preconditioning, and achieve very fast convergence.
Since the diffusion properties of the foreground and background of different images may vary, the authors consider separate segmentations for the detected foreground only-areas and background-only areas, respectively
This is done since the segmentation with respect to one of them could be good but not with respect to the other and combining the results of foreground and background segmentations produces more coherent segmentation and takes advantage of their complementary functions.
The bottom right image shows the solution of the Laplacian propagation, given the initial regions.
After the Laplacian propagation, a stronger separation between foreground and background is obtained.
As seen later in the experiments, even partial segmentations are helpful and the method offers improvement in performance

## Fine-grained recognition with segmentation
This section describes how the authors use the segmented image in the final fine-grained recognition task.
One thing to note here is that, because of the decision to apply HOG type features and pooling to the segmented image, the segmentation helps with both providing shape of the contour of the object to be recognized, as well as, ignoring features in the background that can be distractors.
The authors note here that the authors re-extract features from the segmented image and since much ‘cleaner’ local features are extracted at the boundary, they provide very useful signal, pooled globally.
The authors believe this is crucial for the improvements the authors achieved.
The authors' segmentation run-time allows it to be run as a part of standard recognition pipelines at test time, which had not been possible before, and is a significant advantage

![alt text](https://github.com/vasanthgx/review1/blob/main/images/lp6.png)

## Experiments

The authors show experimental results of the proposed algorithm on a number of fine-grained recognition benchmarks: Oxford 102 flowers, Caltech-UCSD 200 birds, and the recent Oxford Cats and Dogs datasets.
In each case the authors report the performance of the baseline classification algorithm, the best known benchmark results achieved on this dataset, and the proposed algorithm in the same settings.
The authors compare to the baseline algorithm, because it measures how much the proposed segmentation has contributed to the improvement in classification performance.
The authors measure the performance on the large-scale 578-category flower dataset

## [Oxford 102 flower species dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
Oxford 102 flowers dataset is a well-known dataset for fine-grained recognition proposed by Nilsback and Zisserman.
The dataset contains 102 species of flowers and a total of 8189 images, each category containing between 40 and 200 images.
It has well established protocols for training and testing, which the authors adopt too.
A lot of methods have been tested on this dataset, including some segmentation-based.
The performance of the approach on this dataset is 80.66%, which outperforms all previous known methods in the literature.
One important thing to note is that the improvement of the algorithm over the baseline is about 4%, and the only difference between the two is the addition of the proposed segmentation algorithm and the features extracted from the segmented image

## [Caltech-UCSD 200 birds species dataset](https://authors.library.caltech.edu/records/cvm3y-5hh21)

Caltech-UCSD-200 Birds dataset is a very challenging dataset containing 200 species of birds.
Apart from very fine-differences between different species of birds, what makes the recognition hard in this dataset is the variety of poses, large variability in scales, and very rich backgrounds in which the birds often blend in.
The best classification performance achieved on this data is 16.2% classification rate by.
Even when using ground truth bounding boxes, provided as annotations with the dataset , the reported results have been around 19% and most recently 24.3% , but the latter result uses crude ground truth segmentation of each bird

## Method
The authors' baseline Nilsback and Zisserman  Ito and Cubota  Nilsback and Zisserman  Chai, Bicos method Chai, BicosMT method  Ours Ours: improvement over the baseline.
The authors' algorithm shows improvement over all known prior approaches, when no ground truth bounding boxes are used
In this case the authors observed 17.5% classification rate compared to previous 15.7% and 16.2%, The authors' baseline algorithm here achieves only 14.4% which in on par with the performance of SPM-type methods in this scenario.
Another thing to notice here is that the improvement over the baseline, when no bounding boxes information is known, is larger than the improvement with bounding boxes.
This underlines the importance of the proposed automatic detection and segmentation of the object, which allows to ‘zoom in’ on the object, especially for largescale datasets for which providing bounding boxes or other ground truth information will be infeasible

## [Oxford Cats and Dogs dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

Oxford Cats and Dogs  is a new dataset for fine-grained classification which contains 6033 images of 37 breeds of cats and dogs.
Parkhi et al, who collected the dataset, showed impressive performance on this dataset.
They apply segmentation at test time, as is done here, but their algorithm is based on Grabcut , The authors' baseline Chai, Bicos segmentation  Chai, BicosMT segmentation  Ours Ours, improvement over the baseline.
The authors compared the performance on this dataset with the prespecified protocol proposed in the paper (Table 4)
For this dataset too, the authors see that the general method outperforms the best category-specific one from them  and is far better than their more general approach or a bag of words-based method.
Note that they reported classification when using cat and dog head annotations or ground truth segmentation during testing, whereas here the experiments do not use such information.

## Large-scale 578 flower species dataset

This dataset consists of 578 species of flowers and contains about 250,000 images and is the largest and most challenging such dataset the authors are aware of.
The authors' baseline Ours Ours, improvement over the baseline top 1 having an improvement of about 4.41%, top 5 of about 2.7% and top 10 of about 2%
Note that this large-scale data has no segmentation ground truth or bounding box information.
Here the advantage that an automatic segmentation algorithm can give in terms of improving the final classification performance is really important
Another interesting fact is that here the authors have used the same initial region detection model that was trained on the Oxford 102 flowers dataset, which contains fewer species of flowers (102 instead of 578).
This was motivated again by the lack of good ground truth for such a large volume of data.
The performance of the segmentation algorithm can be further improved after adapting the segmentation model to this specific dataset.

## Findings

![alt text](https://github.com/vasanthgx/review1/blob/main/images/tables.png)

The authors observed more than a 4% improvement in the recognition performance on a challenging large-scale flower dataset, containing 578 species of flowers and 250,000 images.
The authors' algorithm achieves 30.17% classification performance compared to 19.2  in the same setting, which in an improvement of 11% over the best known baselines in this scenario
Another interesting observation is that the algorithm achieves a performance of 27.60% when applying segmentation alone.
The authors' algorithm shows improvement over all known prior approaches, when no ground truth bounding boxes are used
In this case the authors observed 17.5% classification rate compared to previous 15.7% and 16.2%, The authors' baseline algorithm here achieves only 14.4% which in on par with the performance of SPM-type methods in this scenario.
The authors' baseline Ours Ours, improvement over the baseline top 1 having an improvement of about 4.41%, top 5 of about 2.7% and top 10 of about 2%.

## Discussion

As seen by the improvements over the baseline, the segmentation algorithm gives advantage in recognition performance.
This is true even if the segmentation may be imperfect for some examples.
This shows that segmenting out the object of interest during testing is of crucial importance for an automatic algorithm and that it is worthwhile exploring even better segmentation algorithms.

## Conclusions and future work

The authors propose an algorithm which combines region-based detection of the object of interest and full-object segmentation through propagation.
The segmentation is applied at test time and is shown to be very useful for improving the classification performance on four challenging datasets.
The authors tested the approach on the most contemporary and challenging datasets for fine-grained recognition improved the performances on all of them.
578-category flower dataset which is the largest collection of flower species the authors are aware of.
The improvements in performance over the baseline are about 3-4%, which is consistent across all the experiments.
The authors' algorithm is much faster than previously used segmentation algorithms in similar scenarios, e.g.
It is applicable to a variety of types of categories, as shown on birds, flowers, and cats and dogs.
The authors' future work will consider improvements to the feature model, e.g. represent it as a mixture of sub models, each one responsible for a subset of classes that are very similar to each other but different as a group from the rest.

## References

1.	Farrell R, Oza O, Zhang N, Morariu V, Darrell T, Davis L. Birdlets: Subordinate categorization using volumetric primitives and pose-normalized appearance. In 2011. p. 161–8. 
2.	The Caltech-UCSD Birds-200-2011 Dataset [Internet]. [cited 2024 May 31]. Available from: https://authors.library.caltech.edu/records/cvm3y-5hh21






## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth_1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

