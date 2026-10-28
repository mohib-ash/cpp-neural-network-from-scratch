<div align="center">

# 🧠 C++ Neural Network From Scratch

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-11%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)](https://github.com/)

*A lightweight, vanilla C++ implementation of a feedforward neural network built from scratch—no external machine learning frameworks required.*

</div>

---

## 📌 Overview

This project implements a small feedforward neural network in C++ and trains it to solve the classic **XOR classification problem**.

Rather than relying on high-level libraries such as PyTorch or TensorFlow, the core mathematical operations are implemented directly using standard C++ and the STL. 

---

## 💡 Why I Built This

Some programming languages stay with you, even when your career takes you in a different direction.

C++ was one of my earlier technical interests, and despite spending much of my recent time working with Python, backend engineering, and AI-related systems, my appreciation for C++ never really went away. Python and I have developed a productive professional relationship, but my old love for C++ deserves some attention too.

> **Why this project?**
> C++ was on my resume, but I did not yet have a C++ project that demonstrated how I applied it. Instead of building another conventional console application, I decided to work on something closer to my core focus: Artificial Intelligence.

Neural networks were a natural choice. They allowed me to combine my interest in AI with my desire to return to C++, implementing the learning process directly rather than treating it as a black box.

In other words, I came back to C++ by building something I genuinely enjoy working on: **AI-related engineering, implemented the vanilla way.**

---

## ✨ Technical Features

* **Feedforward Neural Network:** Input, hidden, and output layers with configurable layer sizes.
* **Forward Propagation:** Computes weighted sums, applies biases, and passes activations through the network.
* **Activation Functions:** ReLU for the hidden layer and sigmoid for the output layer.
* **Mean Squared Error (MSE):** Measures prediction error against target values.
* **Backpropagation:** Propagates gradients through the network to calculate parameter updates.
* **Gradient Descent:** Updates weights and biases using a configurable learning rate.
* **Reusable Workspace Buffers:** Reuses vectors for intermediate activations and gradients to reduce repeated allocations in the training loop.
* **XOR Classification:** Trains and evaluates the network on all four XOR input combinations.

---

## 🏗️ Architecture

The network follows a simple feedforward architecture:

```text
Input Layer          Hidden Layer          Output Layer

   [x1]  ───────┐
                ├──────> [h1] ──────┐
   [x2]  ───────┼──────> [h2] ──────┼──────> [y]
                └──────> [h3] ──────┘

       Input → Hidden → Output
```

For the XOR experiment, the network configuration is structured as follows:

<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
  <thead>
    <tr style="background-color: #1f2937; color: #ffffff;">
      <th style="border: 1px solid #374151; padding: 10px; text-align: left;">Component</th>
      <th style="border: 1px solid #374151; padding: 10px; text-align: left;">Configuration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Input Layer</b></td>
      <td style="border: 1px solid #374151; padding: 10px;">2 neurons</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Hidden Layer</b></td>
      <td style="border: 1px solid #374151; padding: 10px;">3 neurons</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Output Layer</b></td>
      <td style="border: 1px solid #374151; padding: 10px;">1 neuron</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Hidden Activation</b></td>
      <td style="border: 1px solid #374151; padding: 10px;"><code>ReLU</code></td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Output Activation</b></td>
      <td style="border: 1px solid #374151; padding: 10px;"><code>Sigmoid</code></td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Loss Function</b></td>
      <td style="border: 1px solid #374151; padding: 10px;">Mean Squared Error (MSE)</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><b>Optimizer</b></td>
      <td style="border: 1px solid #374151; padding: 10px;">Gradient Descent</td>
    </tr>
  </tbody>
</table>

---

## 🧪 XOR Experiment

XOR is a useful introductory test because its output cannot be represented by a single linear decision boundary. A hidden layer allows the network to learn a nonlinear decision function.

### Training Dataset

<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
  <thead>
    <tr style="background-color: #1f2937; color: #ffffff;">
      <th style="border: 1px solid #374151; padding: 10px; text-align: left;">Input (x1, x2)</th>
      <th style="border: 1px solid #374151; padding: 10px; text-align: center;">Expected Output (y)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[0, 0]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">0</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[0, 1]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">1</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[1, 0]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">1</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[1, 1]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">0</td>
    </tr>
  </tbody>
</table>

### Sample Predictions

After training, the network achieved successful convergence with the following evaluation outputs:

<table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
  <thead>
    <tr style="background-color: #1f2937; color: #ffffff;">
      <th style="border: 1px solid #374151; padding: 10px; text-align: left;">Input</th>
      <th style="border: 1px solid #374151; padding: 10px; text-align: center;">Predicted Output</th>
      <th style="border: 1px solid #374151; padding: 10px; text-align: center;">XOR Decision (Threshold = 0.5)</th>
      <th style="border: 1px solid #374151; padding: 10px; text-align: center;">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[0, 0]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><code>0.000170957</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><b>0</b></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">✅ Correct</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[0, 1]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><code>0.999937000</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><b>1</b></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">✅ Correct</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[1, 0]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><code>0.999937000</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><b>1</b></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">✅ Correct</td>
    </tr>
    <tr>
      <td style="border: 1px solid #374151; padding: 10px;"><code>[1, 1]</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><code>0.000170957</code></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;"><b>0</b></td>
      <td style="border: 1px solid #374151; padding: 10px; text-align: center;">✅ Correct</td>
    </tr>
  </tbody>
</table>

> *Note: These results demonstrate successful learning on the XOR dataset as a correctness check, not a claim of generalization to larger unseen datasets.*

---

## 🛠️ Implementation Details

The implementation uses standard C++ library components, primarily:

* `std::vector` for storing layer activations, weights, biases, and gradients.
* References to avoid unnecessary copies when accessing vectors.
* Nested vectors to represent weight matrices.
* Explicit loops for forward propagation, gradient calculations, and parameter updates.
* Reusable intermediate buffers for activations and gradients.

The goal is to keep the learning process visible and understandable, making it easier to inspect how mathematical operations translate into executable code.

---

## 🚀 Building and Running

### Prerequisites

* A C++ compiler supporting **C++11** or later (e.g., `g++`, `clang++`, or MSVC).
* Standard C++ Library.

### Compilation

Save the source file as `main.cpp`, then compile using:

```bash
g++ -std=c++11 -O2 main.cpp -o neural_network
```

### Execution

**Linux / macOS:**
```bash
./neural_network
```

**Windows (PowerShell / CMD):**
```powershell
.\neural_network.exe
```

The program will execute, outputting loss benchmarks periodically alongside the final predictions for each XOR case.

---

## 🎯 Why I Enjoyed Building This

Coming back to C++ after spending so much time in higher-level ecosystems was a great reminder of why I liked it in the first place. This project was a fun exercise in:

* Seeing how modern development patterns and problem-solving habits translate back into raw C++.
* Enjoying the tactile control over memory layout, explicit `std::vector` buffers, and performance choices.
* Spending some quality time with my old tech love, just for the satisfaction of building something cool from scratch one last time.


---


## 📝 Closing Note

This project represents a return to C++ through a subject I genuinely enjoy: **Artificial Intelligence**.

It combines a personal interest in systems-level programming with an exploration of the mathematical machinery behind neural networks, keeping the implementation deliberately simple enough to inspect, understand, and improve.

> *Sometimes the best way to understand an abstraction is to build a small version of it yourself.*


## 📄 License

This project is open-source and available under the [MIT License](LICENSE).