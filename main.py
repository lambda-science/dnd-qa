"""Python file to serve as the frontend"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit_chat import message

from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

st.set_page_config(page_title="D&D 🗡️ Spell QA Bot", page_icon="🗡️")

# Load the LangChain.
system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)


@st.cache_resource
def load_chroma():
    persist_directory = "db_spells"
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return vectordb


vectordb = load_chroma()
chain_type_kwargs = {"prompt": prompt}
chain = VectorDBQAWithSourcesChain.from_chain_type(
    ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    vectorstore=vectordb,
    chain_type_kwargs=chain_type_kwargs,
)


# From here down is all the StreamLit UI.
st.header("D&D 🗡️ Spell QA Bot")
st.markdown(
    """
This is a chatbot that can answer questions about **Dungeon and Dragons spells** based on this [database](https://www.aidedd.org/dnd-filters/spells-5e.php) and built with LangChain and OpenAI API.  
The creator of this bot is **[Corentin Meyer (@corentinm_py)](https://twitter.com/corentinm_py)**.  
Try by yourself by typing something like: "What's the size of tsunami spell ?"
"""
)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input(
        "You: ", "What's the size of tsunami spell ?", key="input"
    )
    return input_text


user_input = get_text()

if user_input:
    result = chain(
        {"question": user_input},
        return_only_outputs=True,
    )
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
