import streamlit as st

from contextlib import contextmanager
from typing import Any, Generator, List, Optional
from langchain.callbacks import get_callback_manager
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.llms.openai import OpenAI
from langchain.schema import Generation, LLMResult
from random import random


#price per token.
DAVINCI_PRICE = 0.020
CURIE_PRICE = 0.0020
BABBAGE_PRICE = 0.0005
ADA_PRICE = 0.0004

class GenieOpenAICallbackHandler(OpenAICallbackHandler):
    instance_id: float = random()
    tokens: dict = {}
    total_cost = 0

    def __del__(self):
        print("GenieOpenAICallbackHandler is destroyed!")

    def __init__(self, tokens, instance_id) -> None:
        super().__init__()
        self.instance_id = instance_id
        self.tokens = tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:

        if response.llm_output is not None:
            if "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
                model_name = response.llm_output["model_name"]

                if self.tokens.get(model_name) is None:
                    self.tokens[model_name] = {
                        "count": 1,
                        "total_tokens": token_usage["total_tokens"],
                        "total_cost": calculate_price(model_name,
                                                      token_usage["total_tokens"])
                    }
                else:
                    self.tokens[model_name] = {
                        "count": self.tokens[model_name]["count"] + 1,
                        "total_tokens":
                            self.tokens[model_name]["total_tokens"] +
                            token_usage["total_tokens"],
                        "total_cost":
                            self.tokens[model_name]["total_cost"] +
                            calculate_price(model_name, token_usage["total_tokens"])
                    }

                if "total_tokens" in token_usage:
                    self.total_cost += calculate_price(model_name,
                                                       token_usage["total_tokens"])
                    self.total_tokens += token_usage["total_tokens"]

def calculate_price(model_name, token_usage):
     '''Calculate price of the model.
     
     Args:
         model_name: The name of the model.
         token_usage: The number of tokens used.
      Calculate the price of the model based of the pricing per token.
      Davinci: $0.00006 per token
      Curie: $0.00032 per token
      Babbage: $0.00065 per token

     Returns:
         The price of the model.
     '''
     if(model_name == "davinci"):
         return token_usage * DAVINCI_PRICE
     elif(model_name == "curie"):
         return token_usage * CURIE_PRICE
     elif(model_name == "babbage"):
          return token_usage * BABBAGE_PRICE
     elif(model_name == "ada"):
          return token_usage * ADA_PRICE
     else:
           return 0
     

    
class GenieOpenAI(OpenAI):
    """Generic OpenAI class that uses model name."""

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.
        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The full LLM output.
        Example:
            .. code-block:: python
                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        if stop is not None:
            if "stop" in params:
                raise ValueError(
                    "`stop` found in both the input and default params.")
            params["stop"] = stop

        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i: i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        choices = []
        token_usage = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for _prompts in sub_prompts:
            response = self.client.create(prompt=_prompts, **params)
            choices.extend(response["choices"])
            _keys_to_use = _keys.intersection(response["usage"])
            for _key in _keys_to_use:
                if _key not in token_usage:
                    token_usage[_key] = response["usage"][_key]
                else:
                    token_usage[_key] += response["usage"][_key]
        generations = []
        for i, prompt in enumerate(prompts):
            sub_choices = choices[i * self.n: (i + 1) * self.n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason"),
                            logprobs=choice.get("logprobs"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        return LLMResult(
            generations=generations, llm_output={
                "token_usage": token_usage, "model_name": self.model_name}
        )


@contextmanager
def get_openai_callback() -> Generator[GenieOpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    handler = GenieOpenAICallbackHandler(dict(), random())
    manager = get_callback_manager()
    manager.add_handler(handler)
    yield handler
    manager.remove_handler(handler)


