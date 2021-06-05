# Bird Classification Challenge

For the final project, we joined the class bird classification challenge. The
problem was to categorize various species of birds. Data was provided by the
competition, and the main technique used was a [Residual
Network](https://arxiv.org/abs/1512.03385). We started with a network using only
34 layers, and then later moved our training code to Google Colab to then use a
network with 152 layers.

Initially when training, we found that the model was overfitting to the data, as
which makes sense as there are only around 50000 images and 555 labels. As a
result, we used data augmentation, specifically the "RandomResizedCrop", and
"RandomHorizontalFlip", as these were both common augmentations that do not
affect the perceived visual categorizations of birds. After using these
augmentations, we achieved our final accuracy of around 0.69.

We implemented all of the training code in this repository ourself. We also
initialized the weights of the networks used from pretrained networks on
Imagenet.

Our project video is [here](https://youtu.be/dLXy_UliSNM)
