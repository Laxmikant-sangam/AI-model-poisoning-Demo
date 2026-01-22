# AI-model-poisoning-Demo
Adversarial AI lab.  Demonstrates how small  changes to input data can  trick a Machine Learning model

**Introduction**

Imagine showing a computer a picture of a Panda. The computer says, "I am 99% sure that is a Panda."

Now, imagine you add a layer of "static noise" to that picture. The noise is so faint that to your eyes, the picture looks exactly the same. But when you show it to the computer again, it confidently says:

"I am 99% sure that is a Gibbon."

This project demonstrates exactly how that happens. We create a "smart" AI (a Convolutional Neural Network) that can read handwritten numbers. Then, we use a mathematical trick called the Fast Gradient Sign Method (FGSM) to attack it, forcing it to make mistakes without changing the image visibly.
# 🧠 AI Model Poisoning Demo: Adversarial Attacks

> **Mini Project:** AI Security & Adversarial Machine Learning  
> **Topic:** Demonstrating how invisible noise can trick a Deep Learning Model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📖 Project Overview
Artificial Intelligence models, specifically Convolutional Neural Networks (CNNs), are powerful but fragile. This project demonstrates an **Adversarial Evasion Attack**. 

We train a model to recognize handwritten digits with high accuracy (99%). Then, using the **Fast Gradient Sign Method (FGSM)**, we mathematically calculate a specific pattern of "noise" (poison). When this noise is added to an image, it becomes invisible to the human eye, yet completely confuses the AI model.

### 💡 The Core Concept (Simple Analogy)
Imagine a security guard dog (The AI). It is trained to recognize intruders.
* **Normal Scenario:** An intruder walks in. The dog barks.
* **Attack Scenario:** The intruder wears a specific perfume (The Noise). To a human, nothing changed. But the dog smells the perfume, gets confused, and thinks the intruder is its owner.
* **Result:** The security system is bypassed.

---

## 🛠️ Tech Stack used
* **Language:** Python 3
* **Core Library:** TensorFlow 2.x (Keras)
* **Mathematics:** NumPy
* **Visualization:** Matplotlib
* **IDE:** Visual Studio Code

---

## ⚙️ How to Run this Project

### Step 1: Clone or Download
Download this repository to your local machine.

### Step 2: Install Dependencies
Open your terminal/command prompt in the project folder and run:
```bash
    pip install -r requirements.txt
    python attack.py
**View Results**
The script will train the model for 3 Epochs (approx 1 minute).

It will output the Clean Prediction (Correct) vs. Attacked Prediction (Wrong) in the terminal.

A window will pop up showing the Original Image, the Noise (amplified), and the Adversarial Image
