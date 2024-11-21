# CS202 - Computer Security - Lab 4 Adversarial Learning - 24 Fall

## Intro

In this Lab, we would learn how to generate adversarial examples to attack machine learning models. More specifically, we will learn how to generate a set of adversarial images to deceive a CNN deep learning model to make them classify incorrecly.

## Prerequisite

Even though it's not necessary, I would suggest you having a machine with nvidia GPU CUDA support, which can accelerate predicting speed. If you don't have one, you can use [Google Colab](https://colab.research.google.com), which provides nvidia T4 for free. 

We will use learning framework `pytorch` and its `torchvision` library. About how to install them locally, referring this [link](https://pytorch.org/get-started/locally/). Install other necessary libraries. 

In addition to this, you still need download two files:

1. [Testset (testset.pt)](https://drive.google.com/file/d/1OtEynauckSwpf7UBYNZsClo6eUQgJegq/view?usp=sharing), which includes 100 images and their corresponding correct labels ((input_imgs, labels)). Each image is a tensor with shape $3\times 32 \times 32$ (32 is the width and height of the image, and each pixel has 3 channels-red, green and blue). Each label is a single integer, which is the label. You need eventually made some minor perturbations to these images.
2. [ResNet Network (resnet.model)](https://drive.google.com/file/d/1oBIgxX-oWbtWB6el-MQt4nxOBSZ6JeCt/view?usp=share_link), which is a classification neural network. It receives an image tensor and will generate an output with size `1000`. Each term in this torch indicates the probability that this image predicted by this model is the corresponsing labels. 

## Algorithms

We will introduce two adversarial training algorithms-fast gradient sign method (FGSM) and projected gradient descent (PGD). You are required to learn FGSM by yourself and generate the adversarial samples using it. PGD is optional and worth 2 bonus points. 

#### Fast Gradient Sign Method (10 pts)

Pytorch provides the tutorial about this algorithm in detail. Just follow the [instruction](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

Generally, your code will contain following parts:

1. Load provided ResNet model and data set. 
2. FGSM algorithm, which receives the model, input images, desirable labels and your choice of $\epsilon$. Then return the adversarial pictures. 
3. Store the results for all 100 pictures and save them using `torch.save()`

Tips:

The models and data set we provide is definitely **SAFE**, which means there is no executable code inside it. If you have concern, you can load them using `torch.load(weights_only=True)`. However, you need to add a lot of classes into `safe_globals` to load `ResNet` as it depends on a lot of classes. 

```python
# Choose your device
device = device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Load model
model = torch.load(RESNET_MODEL_FILE, map_location=device)
# Load dataset, set the map_location by your preference
testset = torch.load(TESTSET_FILE, weights_only=True)
```

Then you can use `torch.utils.data.DataLoader` to iterate over the dataset or just simply use `for` loop in python.

You can use `pred_result = model(input_imgs)` to get the prediction result. 

#### Projected Gradient Descent (2 pts)

You can learn the basic idea and principle from [link](https://files.sri.inf.ethz.ch/website/teaching/riai2020/materials/lectures/LECTURE3_ATTACKS.pdf).

It's quite similar to FSGM, just in an iterative way. 

## Submission

You need to store all your adversarial samples into a list. Therefore, the finally submitted object should be `List[torch.tensor]` with $\text{length}=100$, containing 100 adversarial images (the order should keep the same as the `testset`). Each tensor has the size of $3\times 32\times 32$. Persist this list using `torch.save(adv_imgs, SAVE_FILE)`. 

Submit your file at [**SUBMISSION SITE**](https://ucr-cs255-24fall-lab4.tch.gdjs2.cn). Your result would be **AUTO-GRADED**. Therefore, if you submit an incorrect file, you would receive low or zero grades. Double check your file before submitting and you can also write your self-check script to test your result. 

## Grade Policy

Our test meets several guidelines:

1. Less $\epsilon$, higher grade. If you make less perturbation to the original image, you would receive a higher grade. If you increase your $\epsilon$, your modification will then be easier to see. Your grade will drop exponentially with $\epsilon$.
2. Higher predicted difference, higher grade. If you acheive a higher mis-prediction rate on your adversarial samples, you would receive a higher grade. Your grade will increase linearly with the mis-prediction rate.

Specifically, we would calculate three values for your ddversarial sampls:

1. $\epsilon$, which is the average difference between your submitted samples and original images (unit in pixel). In FGSM, this $\epsilon$ shoud be the same as your `eps`. In PGD, $\epsilon$ should be lower than `eps` you set in the algorithm as `eps` is the upperbound. 
2. `tp1-accuracy-diff`: 
   1. Top-1 accuracy is a metric used to evaluate the performance of a model. Given an image, the model outputs the probabilities for 1,000 labels. The label with the highest probability is referred to as the top-1 prediction. If this prediction matches the actual label, it is considered a True Positive. Using this, we can calculate the accuracy as $\text{tp1-acc} = \frac{TP}{100}$.
   2. We will calcualate the top-1 accuracy for both original dataset and your dataset and use $|\text{tp1-accuray-original} - \text{tp1-accuracy-yours}|$ as `tp1_accuracy_diff`. A higher Top-1 accuracy difference means that your adversarial examples make it more difficult for the model to produce correct predictions, indicating that your adversarial learning algorithm is more successful.
3. `tp5-accuracy-diff`:
   1. Top-1 accuracy is essentially the same as accuracy. It's somehow too strict. You can turn out picking the most 5 highest prediction values from the probability instead of the highest one, and if any one of them matches the desirable label, we count them as a true positive, we can get Top-5 accuracy. 
   2. We would also calculate the Top-5 accuracy difference for our grading. 

We will use following formula to calculate your grades:

For FGSM, you will get the score according the maximum grade you get. 

$$
\text{TP1 Score} = 0.33 + 33\times \text{tp1-accuracy-diff} - e^{33\times eps}\\
\text{TP5 Score} = 66\times \text{tp5-accuracy-diff} - e^{33\times eps}\\
\text{Final Score} = \text{min}(\text{max}(\text{max}(\text{TP1 Score}, \text{TP5 Score}), 0), 10)
$$

For PGD, you will get the score according to the minimum grade you get.

$$
\text{TP1 Score} = 26\times \text{tp1-accuracy-diff} - e^{66\times eps}\\
\text{TP5 Score} = 46\times \text{tp5-accuracy-diff} - e^{66\times eps}\\
\text{Final Score} = 0.2\times \text{min}(\text{max}(\text{min}(\text{TP1 Score}, \text{TP5 Score}), 0), 10)
$$

PGD algorithm is more powerful and you can see it's more strict to $\epsilon$. 

You can submit for FGSM for **10 TIMES** amd **3 TIMES** for PGD. Test by yourself before submitting.

## Questions

Contacting me via [email](mailto:zxiao033@ucr.edu) if you have any question. Attach your email, name and corresponding submission id. 

**BONUS:** Because this is a security course and you have learned about web security. If you can find any vulnerabilty in this website, we can give you some bonus for the lab :). 

Good luck, have fun!