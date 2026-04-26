# 🕰️ A Beginner's Guide to the Continuous Thought Machine

Welcome! If you've arrived here, you might be curious about this repository and what the **Continuous Thought Machine (CTM)** actually is. Don't worry if the technical details seem overwhelming at first. This guide is designed to explain the core concepts of the CTM in a way that is easy to understand, even if you are not an expert in artificial intelligence.

## What is this repository?
This repository contains the code for a new type of Artificial Intelligence (AI) model created by researchers at Sakana AI.

Imagine how a human thinks before making a decision or taking an action. We often take time to ponder, consider different options, and let our thoughts "unfold" over time. Most traditional AI models don't work like this. They usually process information and spit out an answer in one big leap, almost like a reflex.

The **Continuous Thought Machine** is different. It is an AI model designed to *take its time to think*.

## How does the Continuous Thought Machine work?

At its core, the CTM introduces three main new ideas to make AI "think" more deeply:

1. **An Internal Space for Thought:**
   Think of this like a mental scratchpad. While traditional AIs process input (like an image) directly into output (like a label), the CTM takes the input and puts it into an internal "thought space". It then lets the information simmer and evolve over multiple steps, entirely disconnected from the outside world, before making a final decision. This is like pondering a puzzle in your head before moving a piece.

2. **Neuron-Level Memory (Private Brain Cells):**
   In human brains, cells called neurons process information. The CTM gives each of its artificial "neurons" its own tiny brain and memory. Instead of just reacting to what is happening *right now*, each neuron remembers what has happened in the recent past and uses a small, private set of rules to decide what to do next. It's like having a team of experts, each with their own unique way of looking at a problem's history.

3. **Communicating Through Rhythm (Synchronisation):**
   How do these private neurons share information and make a final decision? Instead of just sending simple signals to each other, they synchronize their rhythms. Imagine a choir where different singers align their timing to create harmony. In the CTM, when the rhythms of different neurons sync up over time, the model uses those synced-up patterns to understand the data and take action.

## What can it do?

Because of this unique ability to "think" over time, the CTM is surprisingly versatile. We have shown that it can solve a wide variety of tasks, some of which are very challenging for traditional AI models:

*   **Classifying Images:** Recognizing objects in pictures (like distinguishing a cat from a dog).
*   **Solving Mazes:** Figuring out the path from a start point to an end point in complex 2D mazes.
*   **Sorting Numbers:** Taking a scrambled list of numbers and putting them in the right order.
*   **Playing Games:** Learning to interact with environments, like keeping a pole balanced on a moving cart (a common test called Reinforcement Learning).

## How can I use this code?

This repository is primarily aimed at researchers and developers who want to experiment with our model.

If you want to play with the code:
1.  **Look in the `tasks` folder:** This is the heart of the project. It's divided into different folders for each of the things the CTM can do (e.g., `mazes`, `image_classification`). Inside each folder, you'll find the specific instructions on how to train the model for that task.
2.  **Try the Tutorial:** We have a notebook file called `examples/01_mnist.ipynb` that provides a step-by-step tutorial on a simple task. This is a great place to start writing code!

If you just want to see it in action without looking at code:
*   **Visit our Interactive Website:** We have a very cool website at [https://pub.sakana.ai/ctm](https://pub.sakana.ai/ctm) where you can play with a live demo of the model solving mazes and see videos of how it works.

We hope this guide helps you understand the exciting ideas behind the Continuous Thought Machine!
