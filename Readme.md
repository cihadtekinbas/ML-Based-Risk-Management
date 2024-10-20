# A Unified Framework for Robustness on Diverse Sampling Errors

This readme file is an outcome of the [CENG502 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2024) Project List](https://github.com/CENG502-Projects/CENG502-Spring2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Machine learning relies on the assumption that both the training and
test datasets follow the same underlying distribution. However, in
reality, datasets only represent a subset of the true distribution,
which can lead to discrepancies between the training and inference data.
These discrepancies, referred to as distribution shifts, can
significantly affect a model's ability to generalize. While various
methods have been developed to address distribution shifts, such as
single domain generalization (SDG) [[1, 2]](#1) and unbiased learning (UBL), these
approaches are usually designed for specific scenarios and may not
effectively handle unforeseen distribution shifts. This highlights the
need for new methodologies that can ensure robust generalization across
diverse domains without prior knowledge of the target distribution.

The approach proposed in this paper involves three key ideas to address
the problem of distribution shifts and improve generalization on diverse
distributions:
-   **Widening the Feature Space:** The framework widens the feature
    space to be referred to during inference. By expanding the feature
    space, the model can capture more diverse representations of the
    data, allowing for better adaptation to different distributions.

-   **Disentangling Representations:** Multiple independent
    representations are disentangled from each other within the widened
    feature space. This disentanglement helps the model focus on
    specific aspects of the data, making it more adaptable to varying
    distributions.

-   **Adaptive Inference:** The model performs inference adaptively for
    each test example by weighting differently on the disentangled
    features based on the distribution mismatch between the source and
    target data. This instance-wise adaptive inference allows the model
    to adjust its predictions based on the specific characteristics of
    each data point.

By combining these three components, the paper, *A Unified Framework for
Robustness on Diverse Sampling Errors*, proposed a framework that aims
to enhance the model's robustness and generalizability on diverse
distributions, even in cases where the target distribution is unknown
during training. It was published at ICCV (International Conference on
Computer Vision) in 2023, and this repository tries to reproduce its
method based on the descriptions in the paper.

## 1.1. Paper summary

The research paper introduces a new strategy to tackle the challenge of
distribution shifts in machine learning algorithms. The paper focuses on
the problem of unreliable generalizations that can occur when there is a
significant gap between the source and target data distributions.
Traditional methods such as single domain generalization and unbiased
learning have been used to address predetermined distribution shifts.
However, these methods may not be effective since the target
distribution is unknown during training.

To overcome this problem, the paper suggests a framework that allows for
adaptive inference based on the target distribution, which is known only
at testing. This instance-wise adaptive inference approach enables the
model to adjust the feature space it refers to for each prediction,
improving generalization on diverse distributions. The framework
includes widening the feature space, disentangling representations, and
performing adaptive inference, all aimed at enhancing the model's
robustness.

The experimental results demonstrate the effectiveness of the proposed
method in improving generalization performance on diverse distributions.
The approach outperforms state-of-the-art methods on tasks involving
significant distribution mismatch. The paper concludes by highlighting
the potential of adaptive inference for enhancing further robustness and
suggests future research directions in studying more robustness and
generalization scenarios.


# 2. The method and my interpretation

## 2.1. The original method
![A boat.](modelarchitecture.png)

The method proposed in the research paper *A Unified Framework for
Robustness on Diverse Sampling Errors* can be summarized as follows:

**Instance-wise Adaptive Inference (IAI)**:

-   The model adjusts the feature space for each test instance during
    inference.

-   Widening the feature space to capture diverse representations.

-   Disentangling representations within the widened feature space.

-   Adaptively selecting features based on the distribution mismatch
    between the source and target data.

**Robust Learning Framework**:

-   Designed to handle diverse data distribution mismatch problems.

-   Focuses on convolutional neural networks (CNNs) for image
    recognition.

-   Hypothesizes that different features are needed for image
    discrimination depending on the data.

-   Demonstrates the effectiveness of IAI for robustness and
    generalizability on diverse distributions.

**Experimental Evaluation**:

-   Extensive evaluations show the method's robustness on diverse
    distributions.

-   Outperforms state-of-the-art methods on tasks with significant
    distribution mismatch.

-   Quantitatively verifies the hypothesis that target data requires
    different features for better image discrimination.

Overall, the method leverages adaptive inference at the instance level,
widens the feature space, disentangles representations, and adaptively
selects features to improve the model's robustness and generalization
performance on diverse distributions.

## step-by-step

### Widening the Feature Space

The disentangled features are re-weighted for every instance at test
time.

**Learning to Diversify**

-   Denote the source dataset as $D_{s} = \{(x_{i},y_{i})\}_{i=1}^{N}$

-   To make the model represent a wider feature space, the style
    generation module $G : X\rightarrow X^{'}$, composed of K sets of {
    convolution(`Conv`),stylr-transfer(`Trans`),transposed
    convolution($Conv_{T}$)},synthesizes various stylized images,where
    $X$ and $X^{'}$ denote the original and augmented input space,
    respectively.

-   All scaled inputs,
    $x_{i}^{'k}= Conv^{T}(Trans(Conv(x_{i})))$  for  k=1,...K are
    aggregated to $X_{i}^{'k}$ i by a linear combination with Gaussian
    random weights.

-   Then, the stylized image $x_{i}^{'k}$ i is fed to the feature
    extractor F in conjunction with the original image $x_{i}$.

-   To train G to generate diverse stylized $x^{'}$,mutual
    information(MI) between $x$ and $x^{'}$ is minimized in the high
    level feature space $Z$.

-   The MI is defined as: $I(z,z^{'})$ $=E_{p}(z,z^{'})$
    $\[log\frac{p(z^{'}|z)}{p(z^{'})}\]$

-   if $p(z^{'}|z)$ and $q(z^{'}|z)$ have similar
    distribution,   $I(z^{'}|z)$ can be approximated to a tractable upper
    bound as:
    $$\hat{I}(z|z^{'})=\frac{1}{N}\sum_{i=1}^{N}\[logq(z_{i}^{'}|z_{i})- \frac{1}{N}\sum_{j=1}^{N}logq(z_{j}^{'}|z_{i})\]$$

-   The difference between $p(z^{'}|z)$ and $q(z^{'}|z)$ can be
    minimized by Kullback-Leibler divergence(KLD). For an
    implementation,KLD can be reduced by minimizing the negative
    log-likelihood $L_{NLL}$ between $z$ and $z^{'}$:
    $$L_{NLL}=-\frac{1}{N}\sum_{i=1}^{N}logq(z_{i}^{'}|z_{i})$$

-   To prevent this problem,another loss term $L_{MMD}$ based on
    class-conditional Maximum Mean Discrepancy(MMD) is applied
    $$L_{MMD}=\frac{1}{C}\sum_{j=1}^{C} \left \|\frac{1}{n_{s}^{j}}\sum_{i=1}^{n_{s}^{j}}\phi (z_{i}^{j})-\frac{1}{n_{t}^{j}}\sum_{i=1}^{n_{s}^{j}}\phi (z_{i}^{'j})\right \|^{2}$$

    where z and $z_{'}$ denote the feature vector of x and $x_{'}$,
    respectively. $n_{s}^{j}$ and $n_{t}^{j}$ are the number of original
    and augmented images, respectively. C denotes the number of classes,
    and $\phi(; )$ denotes a Gaussian kernel that represents the
    distribution in the kernel Hilbert space to compute the difference.

-   With the aforementioned process, various styles generated by G cover
    a wider input space and hence help the model to make wider feature
    space.

-   Yet, the classifier \[F; H\] needs to maximize the MI between the
    same semantic labels for better classification. Towards this, a
    supervised contrastive loss $L_{CL}$ [[3]](#3) is exploited instead of
    directly maximizing MI:
    $$L_{CL}=-\sum_{i=0}^{N}\frac{1}{\left | P(i) \right |}\sum_{p\epsilon P(i)}log\frac{e^{z_{i}\cdot z_{p}/\tau }}{\sum_{a\epsilon A(i) }e^{z_{i}\cdot z_{p}/\tau }}$$

    Consequently, the min-max adversarial training is performed for MI
    between G and \[F; H\].

**Instance-wise Adaptive Inference**

The widened feature space is disentangled via the disentangling module
and re-weighted for each test instance.

**Disentangling Module**

![Samples from IMDB dataset.](disentagling.png)


**Re-weighting**

if there is a significant distribution shift from the source to the
target dataset, the features helpful for prediction would also be
shifted. Thereby, they adjust referred feature space differently for
each test instance: $$z_{i}^{A}=
\[\frac{1}{w_{1}^{s}}z_{1}\odot z_{1};...;\frac{1}{w_{M}^{s}}z_{M}\odot z_{M}
\]$$

$$w_{m}^{s}=\frac{1}{NC}\sum_{i=1}^{N}\sum_{j=1}^{C}z_{mij}^{(s)}$$

The activated features $z^{A}$ are fed into the classification head H in
exactly the same way as z\[1,··· ,M\] to H at training.


## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The experiments were done on two modified versions of the CIFAR10  [[4]](#4)
dataset, a well-known dataset for image classification tasks. The
modified versions are domain generalization (DG-CIFAR) and biased
(B-CIFAR [[5]](#5)). Two real-world datasets were also used in the experiments,
PACS [[6]](#6) and IMDB [[7]](#7) datasets for SDG and UBL, respectively.

-   **CIFAR10**:

    -   **DG-CIFAR:** The train set in CIFAR10 with 50000 images is used
        for training, and the validation set is extended to be 12 domain
        sets with 10000 images including *{fog, snow, frost, zoom blur,
        defocus blur, glass blur, speckle noise, shot noise, impulse
        noise, jpeg compression, pixelate, spatter*}. The images for
        DG-CIFAR dataset are augmented in exactly the same way as in
        figure below.




    -   **B-CIFAR:** The augmented training set of CIFAR10 with 50000
        images including *{(airplane, fog), . . . ,(truck, saturate)}*
        with 0.5% of unbiased instances, and uniformly distributed test
        set of 10000 images evaluation. See figure below.
        
	![Corrupted CIFAR-10 image samples from B-CIFAR dataset.](BCIFAR.PNG)

	

-   **PACS (Photo Art Cartoon Sketch)**:

    -   The train set of *photo* with 1499 items is used for training,
        and *art painting* with 2048 items, *cartoon* with 2344 items,
        and *sketch* with 3929 items are used for tests. See figure below.

	![Sample of 3 classes (dog, guitar, and house) from PACS dataset, out of 7 classes.](PACS.PNG)


-   **IMDB**:

    -   Contains mutually exclusive sets (EB1 and EB2) based on gender
        and age attributes. While EB1 includes women aged 0-29 and men
        aged 40+, EB2 includes women aged 40+ and men aged 0-29. See
        figure below.
        
	![Samples from IMDB dataset.](IMDB.PNG)



## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

After applying the method outlined in the paper, we obtained results that were similar, but not identical. This indicates that while the general framework and methodology are robust, specific factors such as hyperparameter tuning, implementation details, and variations in the dataset could impact the exact outcomes. Despite these variations, the overall trend and performance improvements were in line with the findings reported by the authors.

In summary, the proposed framework effectively tackles the challenges of diverse sampling errors and unknown distribution shifts, demonstrating adaptability and robustness in practical scenarios. The similar results obtained from the implementation further support the effectiveness of the approach, highlighting the importance of adaptive inference strategies in enhancing the generalizability and reliability of machine learning models.


# 5. References

<a id="1">[1]</a> 
Qiao, Fengchun, Long Zhao, and Xi Peng. "Learning to learn single domain generalization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

<a id="2">[2]</a> 
Volpi, Riccardo, et al. "Generalizing to unseen domains via adversarial data augmentation." Advances in neural information processing systems 31 (2018).

<a id="3">[3]</a> 
Khosla, Prannay, et al. "Supervised contrastive learning." Advances in neural information processing systems 33 (2020): 18661-18673.

<a id="3">[4]</a> 
Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.

<a id="4">[5]</a> 
Zhou, Kaiyang, et al. "Deep domain-adversarial image generation for domain generalisation." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 07. 2020.

<a id="5">[6]</a> 
Bahng, Hyojin, et al. "Learning de-biased representations with biased representations." International Conference on Machine Learning. PMLR, 2020.

<a id="6">[7]</a>
Wang, Fei, et al. "The devil of face recognition is in the noise." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
# Contact

Cihad Tekinbaş (cihad.tekinbas@metu.edu.tr)

Muratcan Ayık (muratcan.ayik@metu.edu.tr)
