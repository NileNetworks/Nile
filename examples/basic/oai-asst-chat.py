"""
The most basic chatbot example, using an OpenAssistant agent,
powered by the OpenAI Assistant API.

Run like this:

python3 examples/basic/oai-asst-chat.py
"""

import typer
from rich import print
from rich.prompt import Prompt
from dotenv import load_dotenv

from Nilenetworks.agent.openai_assistant import OpenAIAssistant, OpenAIAssistantConfig
from Nilenetworks.agent.task import Task
from Nilenetworks.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from Nilenetworks.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


@app.command()
def chat() -> None:
    print(
        """
        [blue]Welcome to the basic chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    default_sys_msg = "You are a helpful assistant. Be concise in your answers."

    sys_msg = Prompt.ask(
        "[blue]Tell me who I am. Hit Enter for default, or type your own\n",
        default=default_sys_msg,
    )

    config = OpenAIAssistantConfig(
        system_message=sys_msg,
        llm=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4),  # or GPT4o
    )
    agent = OpenAIAssistant(config)
    task = Task(agent)

    task.run()


if __name__ == "__main__":
    app()
