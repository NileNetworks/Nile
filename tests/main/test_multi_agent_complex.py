import pytest

from Nilenetworks.agent.chat_agent import ChatAgent, ChatAgentConfig
from Nilenetworks.agent.task import Task
from Nilenetworks.agent.tools.orchestration import DoneTool
from Nilenetworks.agent.tools.recipient_tool import RecipientTool
from Nilenetworks.cachedb.redis_cachedb import RedisCacheConfig
from Nilenetworks.language_models.openai_gpt import OpenAIGPTConfig
from Nilenetworks.mytypes import Entity
from Nilenetworks.parsing.parser import ParsingConfig
from Nilenetworks.prompts.prompts_config import PromptsConfig
from Nilenetworks.utils.configuration import Settings, set_global
from Nilenetworks.utils.constants import DONE
from Nilenetworks.vector_store.base import VectorStoreConfig


class _TestChatAgentConfig(ChatAgentConfig):
    max_tokens: int = 200
    vecdb: VectorStoreConfig = None
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        cache_config=RedisCacheConfig(fake=False),
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig()
    prompts: PromptsConfig = PromptsConfig(
        max_tokens=200,
    )


EXPONENTIALS = "3**4 8**3"


@pytest.mark.parametrize("fn_api", [False, True])
@pytest.mark.parametrize("tools_api", [True, False])
@pytest.mark.parametrize("use_done_tool", [True, False])
@pytest.mark.parametrize("constrain_recipients", [True, False])
def test_agents_with_recipient(
    test_settings: Settings,
    fn_api: bool,
    tools_api: bool,
    use_done_tool: bool,
    constrain_recipients: bool,
):
    set_global(test_settings)
    master_cfg = _TestChatAgentConfig(name="Master")

    planner_cfg = _TestChatAgentConfig(
        name="Planner",
        use_tools=not fn_api,
        use_functions_api=fn_api,
        use_tools_api=tools_api,
    )

    multiplier_cfg = _TestChatAgentConfig(name="Multiplier")
    done_tool_name = DoneTool.default_value("request")
    # master asks a series of exponential questions, e.g. 3^6, 8^5, etc.
    if use_done_tool:
        done_response = f"""
            summarize the answers using the TOOL: `{done_tool_name}` with `content` 
            field equal to a string containing the answers without commas,   
            e.g. "243 512 729 125".
        """
    else:
        done_response = f"""
            simply say "{DONE}:" followed by the answers without commas, 
            e.g. "{DONE}: 243 512 729 125".
        """

    master = ChatAgent(master_cfg)
    master.enable_message(DoneTool)
    task_master = Task(
        master,
        interactive=False,
        system_message=f"""
                Your job is to ask me EXACTLY this series of exponential questions:
                {EXPONENTIALS}
                Simply present the needed computation, one at a time, 
                using only numbers and the exponential operator "**".
                Say nothing else, only the numerical operation.
                When you receive the answer, say RIGHT or WRONG, and ask 
                the next exponential question, e.g.: "RIGHT 8**2".
                When done asking the series of questions, 
                {done_response}
                """,
        user_message="Start by asking me an exponential question.",
    )

    # For a given exponential computation, plans a sequence of multiplications.
    planner = ChatAgent(planner_cfg)

    if constrain_recipients:
        planner.enable_message(
            RecipientTool.create(recipients=["Master", "Multiplier"])
        )
    else:
        planner.enable_message(RecipientTool)

    task_planner = Task(
        planner,
        interactive=False,
        system_message="""
                From "Master", you will receive an exponential to compute, 
                but you do not know how to multiply. You have a helper called 
                "Multiplier" who can compute multiplications. So to calculate the
                exponential you receive from "Master", you have to ask a sequence of
                multiplication questions to "Multiplier", to figure out the 
                exponential, remember to use the the TOOL/Function `recipient_message` 
                to ADDRESS the Multipler.
                
                When you have your final answer, report your answer
                back to "Master", ADDRESSING them using the TOOL/Function 
                `recipient_message`.
                
                When asking the Multiplier, remember to only present your 
                request in arithmetic notation, e.g. "3*5"; do not add 
                un-necessary phrases.
                """,
    )

    # Given a multiplication, returns the answer.
    multiplier = ChatAgent(multiplier_cfg)
    task_multiplier = Task(
        multiplier,
        done_if_response=[Entity.LLM],
        interactive=False,
        system_message="""
                You are a calculator. You will be given a multiplication problem. 
                You simply reply with the answer, say nothing else.
                """,
    )

    # planner helps master...
    task_master.add_sub_task(task_planner)
    # multiplier helps planner, but use Validator to ensure
    # recipient is specified via TO[recipient], and if not
    # then the validator will ask for clarification
    task_planner.add_sub_task(task_multiplier)

    result = task_master.run()

    answers = [str(eval(e)) for e in EXPONENTIALS.split()]
    assert all(a in result.content for a in answers)
    # TODO assertions on message history of each agent
