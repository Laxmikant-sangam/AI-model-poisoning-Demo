# AI-model-poisoning-Demo
Adversarial AI lab.  Demonstrates how small  changes to input data can  trick a Machine Learning model

**Introduction**

Imagine showing a computer a picture of a Panda. The computer says, "I am 99% sure that is a Panda."

Now, imagine you add a layer of "static noise" to that picture. The noise is so faint that to your eyes, the picture looks exactly the same. But when you show it to the computer again, it confidently says:

"I am 99% sure that is a Gibbon."

This project demonstrates exactly how that happens. We create a "smart" AI (a Convolutional Neural Network) that can read handwritten numbers. Then, we use a mathematical trick called the Fast Gradient Sign Method (FGSM) to attack it, forcing it to make mistakes without changing the image visibly.
