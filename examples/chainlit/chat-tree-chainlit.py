"""
Variant of chat-tree.py but with Chainlit UI.
The ONLY change is we apply ChainlitTaskCallbacks() to the top-level task!

Run like this:

chainlit run examples/chainlit/chat-tree-chainlit.py
"""

from Nilenetworks.agent.chat_agent import ChatAgent, ChatAgentConfig
from Nilenetworks.agent.task import Task
from Nilenetworks.agent.tool_message import ToolMessage
from Nilenetworks.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from Nilenetworks.utils.globals import GlobalState
from Nilenetworks.utils.configuration import set_global, Settings
from Nilenetworks.utils.constants import DONE
from Nilenetworks.agent.callbacks.chainlit import add_instructions
import Nilenetworks as lr
import chainlit as cl
from textwrap import dedent

INTERACTIVE = False


class MyGlobalState(GlobalState):
    number: int | None = None


class AskNumTool(ToolMessage):
    request = "ask_num"
    purpose = "Ask user for the initial number"


class AddNumTool(ToolMessage):
    request = "add_num"
    purpose = "Add <number> to the original number, return the result"
    number: int

    def handle(self) -> str:
        """
        This is a stateless tool (i.e. does not use any Agent member vars), so we can
        define the handler right here, instead of defining an `add_num`
        method in the agent.
        """
        return str(int(MyGlobalState.get_value("number")) + int(self.number))


class MainChatAgent(ChatAgent):
    def ask_num(self, msg: AskNumTool) -> str:
        res = self.callbacks.get_user_response(prompt="Please enter a number")
        # record this in global state, so other agents can access it
        num = res
        MyGlobalState.set_values(number=num)
        return str(num)


@cl.on_chat_start
async def on_start():
    await add_instructions(
        title="Multi-agent chat for tree-structured computation with tools",
        content=dedent(
            """
        This task consists of performing this calculation for a given input number n:
        
        ```python
        def Main(n):
            if n is odd:
                return (3*n+1) + n
            else:
                If n is divisible by 10:
                    return n/10 + n
                else:
                    return n/2 + n
        ```
        
        See details in the [chat-tree.py](https://github.com/Nilenetworks/Nilenetworks/blob/main/examples/basic/chat-tree.py), 
        and the writeup on 
        [Hierarchical Agent Computation](https://Nilenetworks.github.io/Nilenetworks/examples/agent-tree/).
        
        To start the computation, enter a number.  
        """
        ),
    )


@cl.on_message
async def chat(msg: cl.Message) -> None:
    set_global(
        Settings(
            debug=False,
            cache=True,
            stream=True,
        )
    )

    MyGlobalState.set_values(number=int(msg.content))

    config = ChatAgentConfig(
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4o,
        ),
        vecdb=None,
    )

    main_agent = MainChatAgent(config)
    main_task = Task(
        main_agent,
        name="Main",
        interactive=INTERACTIVE,
        system_message="""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        In this case simply write the <number>, say nothing else.
        
        RESULT Message format: RESULT <number>
        In this case simply say "DONE <number>", e.g.:
        DONE 19
        """,
    )

    # Handles only even numbers
    even_agent = ChatAgent(config)
    even_task = Task(
        even_agent,
        name="Even",
        interactive=INTERACTIVE,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if the <number> is odd, say '{DONE}'
        - otherwise, simply write the <number>, say nothing else.
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # handles only even numbers ending in Zero
    evenz_agent = ChatAgent(config)
    evenz_task = Task(
        evenz_agent,
        name="EvenZ",
        interactive=INTERACTIVE,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is even AND divisible by 10, compute n/10 and pass it on,
        - otherwise, say '{DONE}'
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # Handles only even numbers NOT ending in Zero
    even_nz_agent = ChatAgent(config)
    even_nz_task = Task(
        even_nz_agent,
        name="EvenNZ",
        interactive=INTERACTIVE,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is even AND NOT divisible by 10, compute n/2 and pass it on,
        - otherwise, say '{DONE}'
        
        RESULT Message format: RESULT <number>
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    # Handles only odd numbers
    odd_agent = ChatAgent(config)
    odd_task = Task(
        odd_agent,
        name="Odd",
        interactive=INTERACTIVE,
        system_message=f"""
        You will receive two types of messages, to which you will respond as follows:
        
        INPUT Message format: <number>
        - if <number> n is odd, compute n*3+1 and write it.
        - otherwise, say '{DONE}'

        RESULT Message format: RESULT <number>        
        In this case simply write "DONE RESULT <number>", e.g.:
        DONE RESULT 19
        """,
    )

    adder_agent = ChatAgent(config)
    # set up the tools
    adder_agent.enable_message(AddNumTool)
    # main_agent.enable_message(AskNumTool)

    adder_task = Task(
        adder_agent,
        name="Adder",
        interactive=INTERACTIVE,
        system_message="""
        You will be given a number n.
        You have to add it to the original number and return the result.
        You do not know the original number, so you must use the 
        `add_num` tool/function for this. 
        When you receive the result, say "DONE RESULT <result>", e.g.
        DONE RESULT 19
        """,
    )

    # set up tasks and subtasks
    main_task.add_sub_task([even_task, odd_task])
    even_task.add_sub_task([evenz_task, even_nz_task])
    evenz_task.add_sub_task(adder_task)
    even_nz_task.add_sub_task(adder_task)
    odd_task.add_sub_task(adder_task)

    # inject chainlit callbacks: this is the ONLY change to chat-tree.py
    lr.ChainlitTaskCallbacks(main_task)

    # start the chat
    await main_task.run_async(msg.content)
