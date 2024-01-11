# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perplexity (PPL) measures exponentiated average negative log-likelihood of a sequence, which is a common measurement for evaluating language models. Intended for use with Hugging Face evaluate library."""
from typing import List, Optional
import logging
import evaluate
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2024}
}
"""

logger = logging.getLogger(__name__)

# TODO: Add description of the module here
_DESCRIPTION = """\
Perplexity (PPL) measures exponentiated average negative log-likelihood of a sequence, which is a common measurement for evaluating language models. Intended for use with Hugging Face transformer pipeline and evaluate libraries.

Based on Hugging Face perplexity measurement, with improved handling of pipelines. https://huggingface.co/spaces/evaluate-measurement/perplexity
For more information, see https://huggingface.co/docs/transformers/perplexity
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores

Args:
    model (AutoModelForCausalLM): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )
    tokenizer (AutoTokenizer): tokenizer MUST define pad_token if stride size is greater than 1.
    predictions: List text (str) input.
    add_start_token (bool): whether to add the start token to the texts, so the perplexity can include the probability of the first word. Defaults to True.
    stride (int): Strided batch size. See https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in--transformers for example of strided sliding window. Default: 16
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example 1:
        >>> import evaluate

        >>> perplexity = evaluate.load("bitsyai/perplexity", module_type="measurement")
        >>> predictions = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute(model_id="meta-llama/Llama-2-7b-chat-hf"
        ...                              add_start_token=False,
        ...                              predictions=predictions) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 0))
        647.0
        >>> print(round(results["perplexities"][0], 0))
        32.0

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Measurement):
    """TODO: Short description of my evaluation module."""

    def __init__(
        self, model_id: str, device_map="auto", use_fast=True, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            return_tensors="pt",
            add_eos_token=True,
            add_bos_token=True,
            padding="longest",
            padding_side="right",
            use_fast=use_fast,
            trust_remote_code=True,
            device_map=device_map,
        )

    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MeasurementInfo(
            # This is the description that will appear on the modules page.
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            # Homepage of the module for documentation
            # homepage="@TODO",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/bitsyai/huggingface-evaluate-perplexity"],
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed
        pass

    def _compute(
        self,
        predictions: List[str],
        add_start_token: bool = True,
        stride: int = 16,
        max_length: Optional[int] = None,
    ):
        # If the stride window size is larger than 1 token and no pad token is defined, we need a padding token between strided batches.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.warn(
                "Calculating perplexity requires a padding token, but None was set. Setting tokenizer.add_special_tokens({'pad_token': '[PAD]'})"
            )
        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")
        for start_index in evaluate.logging.tqdm(range(0, len(encoded_texts), stride)):
            end_index = min(start_index + stride, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                )
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(bos_tokens_tensor.size(), dtype=torch.int64),
                        attn_mask,
                    ],
                    dim=1,
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
