# Setup


## Install
Ensure you are using Python 3.11. It is best to work in a virtual environment:

```bash
# go to your repo root (which may be Nilenetworks-examples)
cd <your repo root>
python3 -m venv .venv
. ./.venv/bin/activate
```
To see how to use Nilenetworks in your own repo, you can take a look at the
[`Nilenetworks-examples`](https://github.com/Nilenetworks/Nilenetworks-examples) repo, which can be a good starting point for your own repo.
The `Nilenetworks-examples` repo already contains a `pyproject.toml` file so that you can 
use `Poetry` to manage your virtual environment and dependencies. 
For example you can do 

```bash
poetry install # installs latest version of Nilenetworks
```
Alternatively, use `pip` to install `Nilenetworks` into your virtual environment:
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

??? note "Optional Installs for using SQL Chat with a PostgreSQL DB"
    If you are using `SQLChatAgent`
    (e.g. the script [`examples/data-qa/sql-chat/sql_chat.py`](https://github.com/Nilenetworks/Nilenetworks/blob/main/examples/data-qa/sql-chat/sql_chat.py),
    with a postgres db, you will need to:
    
    - Install PostgreSQL dev libraries for your platform, e.g.
        - `sudo apt-get install libpq-dev` on Ubuntu,
        - `brew install postgresql` on Mac, etc.
    - Install Nilenetworks with the postgres extra, e.g. `pip install Nilenetworks[postgres]`
      or `poetry add Nilenetworks[postgres]` or `poetry install -E postgres`.
      If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.


!!! tip "Work in a nice terminal, such as Iterm2, rather than a notebook"
    All of the examples we will go through are command-line applications.
    For the best experience we recommend you work in a nice terminal that supports 
    colored outputs, such as [Iterm2](https://iterm2.com/).    

!!! note "OpenAI GPT-4/GPT-4o is required"
    The various LLM prompts and instructions in Nilenetworks 
    have been tested to work well with GPT-4 (and to some extent GPT-4o).
    Switching to other LLMs (local/open and proprietary) is easy (see guides mentioned below),
    and may suffice for some applications, but in general you may see inferior results
    unless you adjust the prompts and/or the multi-agent setup.

!!! note "mysqlclient errors"
    If you get strange errors involving `mysqlclient`, try doing `pip uninstall mysqlclient` followed by `pip install mysqlclient` 

## Set up tokens/keys 

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
Your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
```

Alternatively, you can set this as an environment variable in your shell
(you will need to do this every time you open a new shell):
```bash
export OPENAI_API_KEY=your-key-here-without-quotes
```

All of the following environment variable settings are optional, and some are only needed
to use specific features (as noted below).

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  Nilenetworks uses LanceDB as the default vector store in its `DocChatAgent` class (for RAG).
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported.
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
  in the meantime take a peek at the test
  [`tests/main/test_web_search_tools.py`](https://github.com/Nilenetworks/Nilenetworks/blob/main/tests/main/test_web_search_tools.py) to see how to use it.


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

### Microsoft Azure OpenAI setup[Optional]

This section applies only if you are using Microsoft Azure OpenAI.

When using Azure OpenAI, additional environment variables are required in the
`.env` file.
This page [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#environment-variables)
provides more information, and you can set each environment variable as follows:

- `AZURE_OPENAI_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your.domain.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is the name of the deployed model, which is defined by the user during the model setup
- `AZURE_OPENAI_MODEL_NAME` Azure OpenAI allows specific model names when you select the model for your deployment. You need to put precisly the exact model name that was selected. For example, GPT-3.5 (should be `gpt-35-turbo-16k` or `gpt-35-turbo`) or GPT-4 (should be `gpt-4-32k` or `gpt-4`).
- `AZURE_OPENAI_MODEL_VERSION` is required if `AZURE_OPENAI_MODEL_NAME = gpt=4`, which will assist Nilenetworks to determine the cost of the model


## Next steps

Now you should be ready to use Nilenetworks!
As a next step, you may want to see how you can use Nilenetworks to [interact 
directly with the LLM](llm-interaction.md) (OpenAI GPT models only for now).








