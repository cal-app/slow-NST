# slow-NST

An implementation of Neural Style Transfer (NST) following the works of Gatys et al. (https://arxiv.org/pdf/1508.06576.pdf).

Neural Style Transfer is a way to generate interesting artwork using a pre-trained convolutional neural network (VGG19 in this case). It does so by extracting the style of one image, and applying it to the content of another image. 

![Example](https://i.imgur.com/XToUxdm.png)

Fundamentally, NST is an optimization procedure that iteratively modifies an image in order to minimize a two-part cost function. Schematically, starting from a noisy image, it will modify this image's pixel values in order to minimize a cost function that encompasses how well it captures the style of the style image, and the content of the content image.

The cost function is comprised of two parts:

The style cost function: "Style" is encoded by the outputs of a selection of hidden convolutional layers in VGG19. For each layer, the cost function computes a Gram matrix of activations across the entire picture. Consequently, style is encoded as a global property (i.e. the model is not sensitive to local variations in style). The cost is then computed by comparing the difference of both Gram matrices for the style and the generated picture.

The content cost function: The content is encoded as the output of a single convolutional layer.

Over each iteration, the model optimize pixel values of the generated image in order to minimize the loss. This means that it has to reach a compromise between the style and content images. Note that it is simple to tweak hyperparameters in order to determine which takes the precedent: for example one might want a heavy stylized image which is not very faithful to the original, or the contrary.

A particular weakness of NST is that it applies stylization in a global manner. I am currently working on implementing the spatial control feature of the follow-up paper (https://arxiv.org/pdf/1611.07865.pdf). This allows to single out which regions are stylized, and one can even use multiple style images, each applied to different regions of the image.
