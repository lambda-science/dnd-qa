# 🗡️D&D Spell QA Bot🗡️

This is a chatbot that can answer questions about **Dungeon and Dragons spells** based on this [database](https://www.aidedd.org/dnd-filters/spells-5e.php) and built with LangChain and OpenAI API. Usefull to find informations quickly instead of browsing through 50 pages of PDF.
The creator of this bot is **[Corentin Meyer (@corentinm_py)](https://twitter.com/corentinm_py)**.  
💪 This bot it based on Notion Question-Answering demo from [LangChain](https://github.com/hwchase17/langchain)

# 🌲 Environment Setup

In order to set your environment up to run the code here, first install all requirements and then launch streamlit app:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

## 🚀 Code to deploy on StreamLit

The code to run the StreamLit app is in `main.py`.
Note that when setting up your StreamLit app you should make sure to add `OPENAI_API_KEY` as a secret environment variable.

## 🧑 Reproduce the embedding and stuff

Run the following command to ingest the data.

```shell
python ingest.py
```

Boom! Now you're done, and you can ask it questions like:

```shell
python qa.py "What's the size of tsunami spell ?"
```
