# Using Nilenetworks with Non-OpenAI LLMs

Nilenetworks was initially written to work with OpenAI models via their API.
This may sound limiting, but fortunately:

- many open-source LLMs can be served via 
OpenAI-compatible endpoints. See the [Local LLM Setup](https://Nilenetworks.github.io/Nilenetworks/tutorials/local-llm-setup/) guide for details.
- there are tools like [LiteLLM](https://github.com/BerriAI/litellm/tree/main/litellm) 
  that provide an OpenAI-like API for _hundreds_ of non-OpenAI LLM providers 
(e.g. Anthropic's Claude, Google's Gemini).
  
Below we show how you can use the LiteLLM library with Nilenetworks.

## Create an `OpenAIGPTConfig` object with `chat_model = "litellm/..."`

!!! note "Install `litellm` extra"
    To use `litellm` you need to install Nilenetworks with the `litellm` extra, e.g.:
    `pip install "Nilenetworks[litellm]"`

Next, look up the instructions in LiteLLM docs for the specific model you are 
interested. Here we take the example of Anthropic's `claude-instant-1` model.
Set up the necessary environment variables as specified in the LiteLLM docs,
e.g. for the `claude-instant-1` model, you will need to set the `ANTHROPIC_API_KEY`
```bash
export ANTHROPIC_API_KEY=my-api-key
```

Now you are ready to create an instance of `OpenAIGPTConfig` with the 
`chat_model` set to `litellm/<model_spec>`, where you should set `model_spec` based on LiteLLM 
docs. For example, for the `claude-instant-1` model, you would set `chat_model` to
`litellm/claude-instant-1`. But if you are using the model via a 3rd party provider,
(e.g. those via Amazon Bedrock), you may also need to have a `provider` part in the `model_spec`, e.g. 
`litellm/bedrock/anthropic.claude-instant-v1`. In general you can see which of
these to use, from the LiteLLM docs.

```python
import Nilenetworks as lr
import Nilenetworks.language_models as lm

llm_config = lm.OpenAIGPTConfig(
    chat_model="litellm/claude-instant-v1",
    chat_context_length=8000, # adjust according to model
)
```

A similar process works for the `Gemini 1.5 Pro` LLM:

- get the API key [here](https://aistudio.google.com/)
- set the `GEMINI_API_KEY` environment variable in your `.env` file or shell
- set `chat_model="litellm/gemini/gemini-1.5-pro-latest"` in the `OpenAIGPTConfig` object

For other gemini models supported by litellm, see [their docs](https://litellm.vercel.app/docs/providers/gemini)

## Gemini LLMs via OpenAI client, without LiteLLM

This is now the recommended way to use Gemini LLMs with Nilenetworks,
where you don't need to use LiteLLM. As of 11/20/2024, these models
are [available via the OpenAI client](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/).

To use Nilenetworks with Gemini LLMs, all you have to do is:

- set the `GEMINI_API_KEY` environment variable in your `.env` file or shell
- set `chat_model="gemini/<model_name>"` in the `OpenAIGPTConfig` object,  
  where <model_name> is one of "gemini-1.5-flash", "gemini-1.5-flash-8b", or "gemini-1.5-pro"

See [here](https://ai.google.dev/gemini-api/docs/models/gemini) for details on Gemini models.

For example, you can use this `llm_config`:

```python
llm_config = lm.OpenAIGPTConfig(
    chat_model="gemini/" + lm.OpenAIChatModel.GEMINI_1_5_FLASH,
)
```

In most tests you can switch to a gemini model, e.g. `--m gemini/gemini-1.5-flash`, 
e.g.:

```bash
pytest -xvs tests/main/test_llm.py --m gemini/gemini-1.5-flash
```

Many of the example scripts allow switching the model using `-m` or `--model`, e.g.

```bash
python3 examples/basic/chat.py -m gemini/gemini-1.5-flash
```



```python

## Working with the created `OpenAIGPTConfig` object

From here you can proceed as usual, creating instances of `OpenAIGPT`,
`ChatAgentConfig`, `ChatAgent` and `Task` object as usual.

E.g. you can create an object of class `OpenAIGPT` (which represents any
LLM with an OpenAI-compatible API) and interact with it directly:
```python
llm = lm.OpenAIGPT(llm_config)
messages = [
    LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM),
    LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
],
response = mdl.chat(messages, max_tokens=50)
```

When you interact directly with the LLM, you are responsible for keeping dialog history.
Also you would often want an LLM to have access to tools/functions and external
data/documents (e.g. vector DB or traditional DB). An Agent class simplifies managing all of these.
For example, you can create an Agent powered by the above LLM, wrap it in a Task and have it
run as an interactive chat app:

```python
agent_config = lr.ChatAgentConfig(llm=llm_config, name="my-llm-agent")
agent = lr.ChatAgent(agent_config)

task = lr.Task(agent, name="my-llm-task")
task.run()
```

## Example: Simple Chat script with a non-OpenAI proprietary model

Many of the Nilenetworks example scripts have a convenient `-m`  flag that lets you
easily switch to a different model. For example, you can run 
the `chat.py` script in the `examples/basic` folder with the 
`litellm/claude-instant-v1` model:
```bash
python3 examples/basic/chat.py -m litellm/claude-instant-1
```

## Quick testing with non-OpenAI models

There are numerous tests in the main [Nilenetworks repo](https://github.com/Nilenetworks/Nilenetworks) that involve
LLMs, and once you setup the dev environment as described in the README of the repo, 
you can run any of those tests (which run against the default GPT4 model) against
local/remote models that are proxied by `liteLLM` (or served locally via the options mentioned above,
such as `oobabooga`, `ollama` or `llama-cpp-python`), using the `--m <model-name>` option,
where `model-name` takes one of the forms above. Some examples of tests are:

```bash
pytest -s tests/test_llm.py --m local/localhost:8000
pytest -s tests/test_llm.py --m litellm/claude-instant-1
```
When the `--m` option is omitted, the default OpenAI GPT4 model is used.

!!! note "`chat_context_length` is not affected by `--m`"
      Be aware that the `--m` only switches the model, but does not affect the `chat_context_length` 
      parameter in the `OpenAIGPTConfig` object. which you may need to adjust for different models.
      So this option is only meant for quickly testing against different models, and not meant as
      a way to switch between models in a production environment.








    