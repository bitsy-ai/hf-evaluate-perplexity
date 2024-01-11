import pytest
import evaluate
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

test_cases = [
    {
        "data": ["lorem ipsum", "Happy Birthday!", "Bienvenue"],
        "meta-llama/Llama-2-7b-chat-hf": {
            "perplexities": [1244.6507568359375, 35.44779586791992, 341.562255859375],
            "mean_perplexity": 540.5536028544108,
        },
    },
]


def test_llama2_perplexity():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    module = evaluate.load("./perplexity.py", model_id=model_id)
    data = test_cases[0]["data"]
    results = module.compute(predictions=data)
    assert results == test_cases[0][model_id]
