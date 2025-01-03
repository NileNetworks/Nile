
![1500x500](https://github.com/user-attachments/assets/9fe9521a-be81-4664-b1c8-23e25a871c15)

`Nilenetworks`
is an intuitive, lightweight, extensible and principled
Python framework to easily build LLM-powered applications, from CMU and UW-Madison researchers. 
You set up Agents, equip them with optional components (LLM, 
vector-store and tools/functions), assign them tasks, and have them 
collaboratively solve a problem by exchanging messages. 

`Nilenetworks` is a fresh take on LLM app-development, where considerable thought has gone 
into simplifying the developer experience; 
it does not use `Langchain`, or any other LLM framework.

>[Nullify](https://www.nullify.ai) uses AI Agents for secure software development. 
> It finds, prioritizes and fixes vulnerabilities. We have internally adapted Nilenetworks's multi-agent orchestration framework in production, after evaluating CrewAI, Autogen, LangChain, Langflow, etc. We found Nilenetworks to be far superior to those frameworks in terms of ease of setup and flexibility. Nilenetworks's Agent and Task abstractions are intuitive, well thought out, and provide a great developer  experience. We wanted the quickest way to get something in production. With other frameworks it would have taken us weeks, but with Nilenetworks we got to good results in minutes. Highly recommended! <br> -- Jacky Wong, Head of AI at Nullify.

# Quick glimpse of coding with Nilenetworks
This is just a teaser; there's much more, like function-calling/tools, 
Multi-Agent Collaboration, Structured Information Extraction, DocChatAgent 
(RAG), SQLChatAgent, non-OpenAI local/remote LLMs, etc. Scroll down or see docs for more.
See the Nilenetworks Quick-Start
that builds up to a 2-agent information-extraction example using the OpenAI ChatCompletion API. 

script showing how you can use Nilenetworks multi-agents and tools
to extract structured information from a document using **only a local LLM**
(Mistral-7b-instruct-v0.2).

```python
import Nilenetworks as lr
import Nilenetworks.language_models as lm

# set up LLM
llm_cfg = lm.OpenAIGPTConfig( # or OpenAIAssistant to use Assistant API 
  # any model served via an OpenAI-compatible API
  chat_model=lm.OpenAIChatModel.GPT4o, # or, e.g., "ollama/mistral"
)
# use LLM directly
mdl = lm.OpenAIGPT(llm_cfg)
response = mdl.chat("What is the capital of Ontario?", max_tokens=10)

# use LLM in an Agent
agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
agent = lr.ChatAgent(agent_cfg)
agent.llm_response("What is the capital of China?") 
response = agent.llm_response("And India?") # maintains conversation state 

# wrap Agent in a Task to run interactive loop with user (or other agents)
task = lr.Task(agent, name="Bot", system_message="You are a helpful assistant")
task.run("Hello") # kick off with user saying "Hello"

# 2-Agent chat loop: Teacher Agent asks questions to Student Agent
teacher_agent = lr.ChatAgent(agent_cfg)
teacher_task = lr.Task(
  teacher_agent, name="Teacher",
  system_message="""
    Ask your student concise numbers questions, and give feedback. 
    Start with a question.
    """
)
student_agent = lr.ChatAgent(agent_cfg)
student_task = lr.Task(
  student_agent, name="Student",
  system_message="Concisely answer the teacher's questions.",
  single_round=True,
)

teacher_task.add_sub_task(student_task)
teacher_task.run()
```

# :gear: Installation and Setup

### Install `Nilenetworks`
Nilenetworks requires Python 3.11+. We recommend using a virtual environment.
Use `pip` to install a bare-bones slim version of `Nilenetworks` (from PyPi) to your virtual 
environment:
```bash
pip install Nilenetworks
```
The core Nilenetworks package lets you use OpenAI Embeddings models via their API. 
If you instead want to use the `sentence-transformers` embedding models from HuggingFace, 
install Nilenetworks like this: 
```bash
pip install "Nilenetworks[hf-embeddings]"
```
For many practical scenarios, you may need additional optional dependencies:
- To use various document-parsers, install Nilenetworks with the `doc-chat` extra:
    ```bash
    pip install "Nilenetworks[doc-chat]"
    ```
- For "chat with databases", use the `db` extra:
    ```bash
    pip install "Nilenetworks[db]"
    ``
- You can specify multiple extras by separating them with commas, e.g.:
    ```bash
    pip install "Nilenetworks[doc-chat,db]"
    ```
- To simply install _all_ optional dependencies, use the `all` extra (but note that this will result in longer load/startup times and a larger install size):
    ```bash
    pip install "Nilenetworks[all]"
    ```
<details>
<summary><b>Optional Installs for using SQL Chat with a PostgreSQL DB </b></summary>

If you are using `SQLChatAgent` 
(e.g. the script [`examples/data-qa/sql-chat/sql_chat.py`](examples/data-qa/sql-chat/sql_chat.py)),
with a postgres db, you will need to:

- Install PostgreSQL dev libraries for your platform, e.g.
  - `sudo apt-get install libpq-dev` on Ubuntu,
  - `brew install postgresql` on Mac, etc.
- Install Nilenetworks with the postgres extra, e.g. `pip install Nilenetworks[postgres]`
  or `poetry add Nilenetworks[postgres]` or `poetry install -E postgres`.
  If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.
</details>

:memo: If you get strange errors involving `mysqlclient`, try doing `pip uninstall mysqlclient` followed by `pip install mysqlclient`.

### Set up environment variables (API keys, etc)

To get started, all you need is an OpenAI API Key.
If you don't have one, see [this OpenAI Page](https://platform.openai.com/docs/quickstart).
(Note that while this is the simplest way to get started, Nilenetworks works with practically any LLM, not just those from OpenAI. 
See the guides to using [Open/Local LLMs](https://Nilenetworks.github.io/Nilenetworks/tutorials/local-llm-setup/), 
and other [non-OpenAI](https://Nilenetworks.github.io/Nilenetworks/tutorials/non-openai-llms/) proprietary LLMs.) 

In the root of the repo, copy the `.env-template` file to a new file `.env`: 
```bash
cp .env-template .env
```
Then insert your OpenAI API Key. 
Your `.env` file should look like this (the organization is optional 
but may be required in some scenarios).
```bash
OPENAI_API_KEY=your-key-here-without-quotes
OPENAI_ORGANIZATION=optionally-your-organization-id
````

Alternatively, you can set this as an environment variable in your shell
(you will need to do this every time you open a new shell):
```bash
export OPENAI_API_KEY=your-key-here-without-quotes
```


<details>
<summary><b>Optional Setup Instructions (click to expand) </b></summary>

All of the following environment variable settings are optional, and some are only needed 
to use specific features (as noted below).

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  Alternatively [Chroma](https://docs.trychroma.com/) or [LanceDB](https://lancedb.com/) are also currently supported. 
  We use the local-storage version of Chroma, so there is no need for an API key.
- **Redis** Password, host, port: This is optional, and only needed to cache LLM API responses
  using Redis Cloud. Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Nilenetworks and even beyond.
  If you don't set up these, Nilenetworks will use a pure-python 
  Redis in-memory cache via the [Fakeredis](https://fakeredis.readthedocs.io/en/latest/) library.
- **Momento** Serverless Caching of LLM API responses (as an alternative to Redis). 
   To use Momento instead of Redis:
  - enter your Momento Token in the `.env` file, as the value of `MOMENTO_AUTH_TOKEN` (see example file below),
  - in the `.env` file set `CACHE_TYPE=momento` (instead of `CACHE_TYPE=redis` which is the default).
- **GitHub** Personal Access Token (required for apps that need to analyze git
  repos; token-based API calls are less rate-limited). See this
  [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- **Google Custom Search API Credentials:** Only needed to enable an Agent to use the `GoogleSearchTool`.
  To use Google Search as an LLM Tool/Plugin/function-call, 
  you'll need to set up 
  [a Google API key](https://developers.google.com/custom-search/v1/introduction#identify_your_application_to_google_with_api_key),
  then [setup a Google Custom Search Engine (CSE) and get the CSE ID](https://developers.google.com/custom-search/docs/tutorial/creatingcse).
  (Documentation for these can be challenging, we suggest asking GPT4 for a step-by-step guide.)
  After obtaining these credentials, store them as values of 
  `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in your `.env` file. 
  Full documentation on using this (and other such "stateless" tools) is coming soon, but 
  in the meantime take a peek at this [chat example](examples/basic/chat-search.py), which 
  shows how you can easily equip an Agent with a `GoogleSearchtool`.
  


If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
CACHE_TYPE=redis # or momento
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
MOMENTO_AUTH_TOKEN=your-momento-token-no-quotes # instead of REDIS* variables
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-cse-id
```
</details>

<details>
<summary><b>Optional setup instructions for Microsoft Azure OpenAI(click to expand)</b></summary> 

When using Azure OpenAI, additional environment variables are required in the 
`.env` file.
This page [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#environment-variables)
provides more information, and you can set each environment variable as follows:

- `AZURE_OPENAI_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your.domain.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is the name of the deployed model, which is defined by the user during the model setup 
- `AZURE_OPENAI_MODEL_NAME` Azure OpenAI allows specific model names when you select the model for your deployment. You need to put precisly the exact model name that was selected. For example, GPT-4 (should be `gpt-4-32k` or `gpt-4`).
- `AZURE_OPENAI_MODEL_VERSION` is required if `AZURE_OPENAI_MODEL_NAME = gpt=4`, which will assist Nilenetworks to determine the cost of the model  
</details>

---

# :whale: Docker Instructions

We provide a containerized version of the [`Nilenetworks-examples`](https://github.com/Nilenetworks/Nilenetworks-examples) 
repository via this [Docker Image](https://hub.docker.com/r/Nilenetworks/Nilenetworks).
All you need to do is set up environment variables in the `.env` file.
Please follow these steps to setup the container:

```bash
# get the .env file template from `Nilenetworks` repo
wget -O .env https://raw.githubusercontent.com/Nilenetworks/Nilenetworks/main/.env-template

# Edit the .env file with your favorite editor (here nano), and remove any un-used settings. E.g. there are "dummy" values like "your-redis-port" etc -- if you are not using them, you MUST remove them.
nano .env

# launch the container
docker run -it --rm  -v ./.env:/Nilenetworks/.env Nilenetworks/Nilenetworks

# Use this command to run any of the scripts in the `examples` directory
python examples/<Path/To/Example.py> 
``` 



# :tada: Usage Examples

These are quick teasers to give a glimpse of what you can do with Nilenetworks
and how your code would look. 

:warning: The code snippets below are intended to give a flavor of the code
and they are **not** complete runnable examples! For that we encourage you to 
consult the [`Nilenetworks-examples`](https://github.com/Nilenetworks/Nilenetworks-examples) 
repository.

:information_source:
The various LLM prompts and instructions in Nilenetworks
have been tested to work well with GPT-4 (and to some extent GPT-4o).
Switching to other LLMs (local/open and proprietary) is easy (see guides mentioned above),
and may suffice for some applications, but in general you may see inferior results
unless you adjust the prompts and/or the multi-agent setup.


:book: Also see the
[`Getting Started Guide`](https://Nilenetworks.github.io/Nilenetworks/quick-start/)
for a detailed tutorial.



Click to expand any of the code examples below.
All of these can be run in a Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nilenetworks/Nilenetworks/blob/main/examples/Nilenetworks_quick_start.ipynb)

<details>
<summary> <b> Direct interaction with OpenAI LLM </b> </summary>

```python
import Nilenetworks.language_models as lm

mdl = lm.OpenAIGPT()

messages = [
  lm.LLMMessage(content="You are a helpful assistant",  role=lm.Role.SYSTEM), 
  lm.LLMMessage(content="What is the capital of Ontario?",  role=lm.Role.USER),
]

response = mdl.chat(messages, max_tokens=200)
print(response.message)
```
</details>

<details>
<summary> <b> Interaction with non-OpenAI LLM (local or remote) </b> </summary>
Local model: if model is served at `http://localhost:8000`:

```python
cfg = lm.OpenAIGPTConfig(
  chat_model="local/localhost:8000", 
  chat_context_length=4096
)
mdl = lm.OpenAIGPT(cfg)
# now interact with it as above, or create an Agent + Task as shown below.
```

If the model is [supported by `liteLLM`](https://docs.litellm.ai/docs/providers), 
then no need to launch the proxy server.
Just set the `chat_model` param above to `litellm/[provider]/[model]`, e.g. 
`litellm/anthropic/claude-instant-1` and use the config object as above.
Note that to use `litellm` you need to install Nilenetworks with the `litellm` extra:
`poetry install -E litellm` or `pip install Nilenetworks[litellm]`.
For remote models, you will typically need to set API Keys etc as environment variables.
You can set those based on the LiteLLM docs. 
If any required environment variables are missing, Nilenetworks gives a helpful error
message indicating which ones are needed.
Note that to use `Nilenetworks` with `litellm` you need to install the `litellm` 
extra, i.e. either `pip install Nilenetworks[litellm]` in your virtual env,
or if you are developing within the `Nilenetworks` repo, 
`poetry install -E litellm`.
```bash
pip install Nilenetworks[litellm]
```
</details>

<details>
<summary> <b> Define an agent, set up a task, and run it </b> </summary>

```python
import Nilenetworks as lr

agent = lr.ChatAgent()

# get response from agent's LLM, and put this in an interactive loop...
# answer = agent.llm_response("What is the capital of Ontario?")
  # ... OR instead, set up a task (which has a built-in loop) and run it
task = lr.Task(agent, name="Bot") 
task.run() # ... a loop seeking response from LLM or User at each turn
```
</details>

<details>
<summary><b> Three communicating agents </b></summary>

A toy numbers game, where when given a number `n`:
- `repeater_task`'s LLM simply returns `n`,
- `even_task`'s LLM returns `n/2` if `n` is even, else says "DO-NOT-KNOW"
- `odd_task`'s LLM returns `3*n+1` if `n` is odd, else says "DO-NOT-KNOW"

Each of these `Task`s automatically configures a default `ChatAgent`.

```python
import Nilenetworks as lr
from Nilenetworks.utils.constants import NO_ANSWER

repeater_task = lr.Task(
    name = "Repeater",
    system_message="""
    Your job is to repeat whatever number you receive.
    """,
    llm_delegate=True, # LLM takes charge of task
    single_round=False, 
)

even_task = lr.Task(
    name = "EvenHandler",
    system_message=f"""
    You will be given a number. 
    If it is even, divide by 2 and say the result, nothing else.
    If it is odd, say {NO_ANSWER}
    """,
    single_round=True,  # task done after 1 step() with valid response
)

odd_task = lr.Task(
    name = "OddHandler",
    system_message=f"""
    You will be given a number n. 
    If it is odd, return (n*3+1), say nothing else. 
    If it is even, say {NO_ANSWER}
    """,
    single_round=True,  # task done after 1 step() with valid response
)
```
Then add the `even_task` and `odd_task` as sub-tasks of `repeater_task`, 
and run the `repeater_task`, kicking it off with a number as input:
```python
repeater_task.add_sub_task([even_task, odd_task])
repeater_task.run("3")
```

</details>

<details>
<summary><b> Simple Tool/Function-calling example </b></summary>

Nilenetworks leverages Pydantic to support OpenAI's
[Function-calling API](https://platform.openai.com/docs/guides/gpt/function-calling)
as well as its own native tools. The benefits are that you don't have to write
any JSON to specify the schema, and also if the LLM hallucinates a malformed
tool syntax, Nilenetworks sends the Pydantic validation error (suitably sanitized) 
to the LLM so it can fix it!

Simple example: Say the agent has a secret list of numbers, 
and we want the LLM to find the smallest number in the list. 
We want to give the LLM a `probe` tool/function which takes a
single number `n` as argument. The tool handler method in the agent
returns how many numbers in its list are at most `n`.

First define the tool using Nilenetworks's `ToolMessage` class:


```python
import Nilenetworks as lr

class ProbeTool(lr.agent.ToolMessage):
  request: str = "probe" # specifies which agent method handles this tool
  purpose: str = """
        To find how many numbers in my list are less than or equal to  
        the <number> you specify.
        """ # description used to instruct the LLM on when/how to use the tool
  number: int  # required argument to the tool
```

Then define a `SpyGameAgent` as a subclass of `ChatAgent`, 
with a method `probe` that handles this tool:

```python
class SpyGameAgent(lr.ChatAgent):
  def __init__(self, config: lr.ChatAgentConfig):
    super().__init__(config)
    self.numbers = [3, 4, 8, 11, 15, 25, 40, 80, 90]

  def probe(self, msg: ProbeTool) -> str:
    # return how many numbers in self.numbers are less or equal to msg.number
    return str(len([n for n in self.numbers if n <= msg.number]))
```

We then instantiate the agent and enable it to use and respond to the tool:

```python
spy_game_agent = SpyGameAgent(
    lr.ChatAgentConfig(
        name="Spy",
        vecdb=None,
        use_tools=False, #  don't use Nilenetworks native tool
        use_functions_api=True, # use OpenAI function-call API
    )
)
spy_game_agent.enable_message(ProbeTool)
```

For a full working example see the
[chat-agent-tool.py](https://github.com/Nilenetworks/Nilenetworks-examples/blob/main/examples/quick-start/chat-agent-tool.py)
script in the `Nilenetworks-examples` repo.
</details>

<details>
<summary> <b>Tool/Function-calling to extract structured information from text </b> </summary>

Suppose you want an agent to extract 
the key terms of a lease, from a lease document, as a nested JSON structure.
First define the desired structure via Pydantic models:

```python
from pydantic import BaseModel
class LeasePeriod(BaseModel):
    start_date: str
    end_date: str


class LeaseFinancials(BaseModel):
    monthly_rent: str
    deposit: str

class Lease(BaseModel):
    period: LeasePeriod
    financials: LeaseFinancials
    address: str
```

Then define the `LeaseMessage` tool as a subclass of Nilenetworks's `ToolMessage`.
Note the tool has a required argument `terms` of type `Lease`:

```python
import Nilenetworks as lr

class LeaseMessage(lr.agent.ToolMessage):
    request: str = "lease_info"
    purpose: str = """
        Collect information about a Commercial Lease.
        """
    terms: Lease
```

Then define a `LeaseExtractorAgent` with a method `lease_info` that handles this tool,
instantiate the agent, and enable it to use and respond to this tool:

```python
class LeaseExtractorAgent(lr.ChatAgent):
    def lease_info(self, message: LeaseMessage) -> str:
        print(
            f"""
        DONE! Successfully extracted Lease Info:
        {message.terms}
        """
        )
        return json.dumps(message.terms.dict())
    
lease_extractor_agent = LeaseExtractorAgent()
lease_extractor_agent.enable_message(LeaseMessage)
```

See the [`chat_multi_extract.py`](https://github.com/Nilenetworks/Nilenetworks-examples/blob/main/examples/docqa/chat_multi_extract.py)
script in the `Nilenetworks-examples` repo for a full working example.
</details>

<details>
<summary><b> Chat with documents (file paths, URLs, etc) </b></summary>

Nilenetworks provides a specialized agent class `DocChatAgent` for this purpose.
It incorporates document sharding, embedding, storage in a vector-DB, 
and retrieval-augmented query-answer generation.
Using this class to chat with a collection of documents is easy.
First create a `DocChatAgentConfig` instance, with a 
`doc_paths` field that specifies the documents to chat with.

```python
import Nilenetworks as lr
from Nilenetworks.agent.special import DocChatAgentConfig, DocChatAgent

config = DocChatAgentConfig(
  doc_paths = [
    "https://en.wikipedia.org/wiki/Language_model",
    "https://en.wikipedia.org/wiki/N-gram_language_model",
    "/path/to/my/notes-on-language-models.txt",
  ],
  vecdb=lr.vector_store.QdrantDBConfig(),
)
```

Then instantiate the `DocChatAgent` (this ingests the docs into the vector-store):

```python
agent = DocChatAgent(config)
```
Then we can either ask the agent one-off questions,
```python
agent.llm_response("What is a language model?")
```
or wrap it in a `Task` and run an interactive loop with the user:
```python
task = lr.Task(agent)
task.run()
```

See full working scripts in the 
[`docqa`](https://github.com/Nilenetworks/Nilenetworks-examples/tree/main/examples/docqa)
folder of the `Nilenetworks-examples` repo.
</details>

<details>
<summary><b> :fire: Chat with tabular data (file paths, URLs, dataframes) </b></summary>

Using Nilenetworks you can set up a `TableChatAgent` with a dataset (file path, URL or dataframe),
and query it. The Agent's LLM generates Pandas code to answer the query, 
via function-calling (or tool/plugin), and the Agent's function-handling method
executes the code and returns the answer.

Here is how you can do this:

```python
import Nilenetworks as lr
from Nilenetworks.agent.special import TableChatAgent, TableChatAgentConfig
```

Set up a `TableChatAgent` for a data file, URL or dataframe
(Ensure the data table has a header row; the delimiter/separator is auto-detected):
```python
dataset =  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# or dataset = "/path/to/my/data.csv"
# or dataset = pd.read_csv("/path/to/my/data.csv")
agent = TableChatAgent(
    config=TableChatAgentConfig(
        data=dataset,
    )
)
```
Set up a task, and ask one-off questions like this: 

```python
task = lr.Task(
  agent, 
  name = "DataAssistant",
  default_human_response="", # to avoid waiting for user input
)
result = task.run(
  "What is the average alcohol content of wines with a quality rating above 7?",
  turns=2 # return after user question, LLM fun-call/tool response, Agent code-exec result
) 
print(result.content)
```
Or alternatively, set up a task and run it in an interactive loop with the user:

```python
task = lr.Task(agent, name="DataAssistant")
task.run()
``` 

For a full working example see the 
[`table_chat.py`](https://github.com/Nilenetworks/Nilenetworks-examples/tree/main/examples/data-qa/table_chat.py)
script in the `Nilenetworks-examples` repo.
