---
title: Perplexity
tags:
- evaluate
- measurement
description: "Perplexity (PPL) measures exponentiated average negative log-likelihood of a sequence, which is a common measurement for evaluating language models."
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
---

# Measurement Card for Perplexity

***Module Card Instructions:*** *Fill out the following subsections. Feel free to take a look at existing measurement cards if you'd like examples.*

## Measurement Description

Based on https://huggingface.co/spaces/evaluate-metric/perplexity/ with improvements to support pipelines. 

Given a model and an input text sequence, perplexity measures how likely the model is to generate the input text sequence.

As a metric, it can be used to evaluate how well the model has learned the distribution of the text it was trained on.

In this case, model_id should be the trained model to be evaluated, and the input texts should be the text that the model was trained on.

This implementation of perplexity is calculated with log base e, as in perplexity = e**(sum(losses) / num_tokenized_tokens), following recent convention in deep learning frameworks.

## How to Use
*Give general statement of how to use the measurement*

*Provide simplest possible example for using the measurement*

### Inputs
*List all input arguments in the format below*
- **input_field** *(type): Definition of input, with explanation if necessary. State any default value(s).*

### Output Values

*Explain what this measurement outputs and provide an example of what the measurement output looks like. Modules should return a dictionary with one or multiple key-value pairs, e.g. {"bleu" : 6.02}*

*State the range of possible values that the measurement's output can take, as well as what in that range is considered good. For example: "This measurement can take on any value between 0 and 100, inclusive. Higher scores are better."*

#### Values from Popular Papers
*Give examples, preferrably with links to leaderboards or publications, to papers that have reported this measurement, along with the values they have reported.*

### Examples
*Give code examples of the measurement being used. Try to include examples that clear up any potential ambiguity left from the measurement description above. If possible, provide a range of examples that show both typical and atypical results, as well as examples where a variety of input parameters are passed.*

## Limitations and Bias
*Note any known limitations or biases that the measurement has, with links and references if possible.*

## Citation
*Cite the source where this measurement was introduced.*

## Further References
*Add any useful further references.*
