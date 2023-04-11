import ast_comments as ast
import openai_async
import asyncio
import logging
import openai
import os
import sys

from rich.progress import Progress

from typing import (
    Any,
    cast,
    Deque,
    DefaultDict,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import typing

logname = "chatdbg.log"
logging.basicConfig(
    filename=logname,
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logging.info("Running ChatDBG.")


async def get_result(
    system_content, user_content, pbar, progress, model="gpt-3.5-turbo", max_trials=3
) -> Optional[openai.api_resources.Completion]:
    import httpx

    try:
        for trial in range(max_trials):
            completion = await openai_async.chat_complete(
                openai.api_key,
                timeout=30,
                payload={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content,
                            "role": "user",
                            "content": user_content,
                        }
                    ],
                },
            )
            code_block = extract_code_block(completion.json())
            logging.info(f"PROCESSING {code_block}")
    except (openai.error.AuthenticationError, httpx.LocalProtocolError):
        print()
        print(
            "You need an OpenAI key to use commentator. You can get a key here: https://openai.com/api/"
        )
        print(
            "Invoke commentator with the api-key argument or set the environment variable OPENAI_API_KEY."
        )
        import sys

        sys.exit(1)
    except Exception as e:
        return ""
    progress.update(pbar, advance=1)
    return code_block


def find_code_start(code: str) -> int:
    """
    Finds the starting location of a code block in a string.

    Args:
        code: A string containing code.

    Returns:
        An integer representing the starting position of the code block.

    """
    lines = code.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    first_line = lines[i].strip()
    if first_line == "```":
        return 3
    if first_line.startswith("```"):
        word = first_line[3:].strip()
        if len(word) > 0 and " " not in word:
            return len(word) + 3
    return -1


def extract_code_block(completion: dict) -> str:
    """
    Extracts code block from the given completion dictionary.

    Args:
        completion (dict): Completion dictionary containing text and other data.

    Returns:
        str: Extracted code block from the completion dictionary.
    """
    c = completion
    text = c["choices"][0]["message"]["content"]
    first_index = find_code_start(text)
    second_index = text.find("```", first_index + 1)
    if first_index == -1 or second_index == -1:
        code_block = text
    else:
        code_block = text[first_index:second_index]
    return code_block
