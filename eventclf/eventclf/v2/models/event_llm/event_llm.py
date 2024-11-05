from enum import Enum
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import pipeline


class LLMModels(str, Enum):
    llama_3_2_1b_instruct = "meta-llama/Llama-3.2-1B-instruct"
    llama_3_2_1b = "meta-llama/Llama-3.2-1B"
    llama_3_2_3b_instruct = "meta-llama/Llama-3.2-3B-instruct"
    llama_3_2_3b = "meta-llama/Llama-3.2-3B"
    llama_3_1_8b_instruct = "meta-llama/Llama-3.1-8B-instruct"
    llama_3_1_8b = "meta-llama/Llama-3.1-8B"


class EventLLM:
    def __init__(
        self,
        model: LLMModels = LLMModels.llama_3_2_1b_instruct,
        batch_size: int = 10,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initilisatises LLM approach for event summary model

        Parameters
        ----------
        model : LLMModels, optional
            Model to use, by default LLMModels.llama_3_2_1b_instruct
        batch_size : int, optional
            number of posts in a batch, by default 10
        device : Union[str,torch.device], optional
            device to use, either GPU or CPU, by default "cpu"
        """
        self._pipe = pipeline(
            "text-generation", model=model, device=device, batch_size=batch_size
        )
        self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id
        self._terminators = [
            self._pipe.tokenizer.eos_token_id,
            self._pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self._pipe.model.generation_config.force_words_ids = [[9642, 2822]]

    def _format_prompt(
        self, prompt_template: str, prompt_inputs: Dict[str], posts: List[str]
    ) -> List[str]:
        """Takes prompt template, additional inputs and posts and formats them to usable input for
        an LLM

        Parameters
        ----------
        prompt_template : str
            template of prompt
        prompt_inputs : Dict[str]
            Dictionary containing additional content to put into prompts
        posts : List[str]
            list of posts to classify

        Returns
        -------
        List[str]
            list of prompt inputs for input to an LLM
        """
        prompt = prompt_template
        for input_key, input_text in prompt_inputs.values():
            regex = "{" + input_key + "}"
            prompt = prompt.replace(regex, input_text)
        prompt_list = [prompt.replace("{post}", post) for post in posts]
        return prompt_list

    def predict(
        self,
        prompt_template: str,
        prompt_inputs: Dict[str],
        posts: List[str],
        temperature: float = 0.1,
        max_new_tokens: int = 1,
    ) -> np.array:
        """Predicts the classification of posts using LLM

        Parameters
        ----------
        prompt_template : str
            template of prompt
        prompt_inputs : Dict[str]
            additional prompt arguments in prompt template
        posts : List[str]
            posts to classify
        temperature : float, optional
            argument that controls the amount of randomness in an LLM, by default 0.1
        max_new_tokens : int, optional
            number of new tokens to return, by default 1

        Returns
        -------
        np.array
            binary array of classifications
        """
        prompt_list = self._format_prompt(
            prompt_template=prompt_template, prompt_inputs=prompt_inputs, posts=posts
        )
        output = self._pipe(
            prompt_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=self._terminators,
        )
        cleaned_output = []
        for prompt, responce in zip(prompt_list, output):
            out = responce[0]["generated_text"][len(prompt) :]
            if "yes" in out.lower():
                cleaned_output.append(1)
            elif "no" in out.lower():
                cleaned_output.append(0)
            else:
                cleaned_output.append(np.NaN)
        return output
