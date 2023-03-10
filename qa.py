"""Ask a question to the database."""
#%%
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import argparse

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

persist_directory = "db_spells"
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
chain_type_kwargs = {"prompt": prompt}
chain = VectorDBQAWithSourcesChain.from_chain_type(
    ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    vectorstore=vectordb,
    chain_type_kwargs=chain_type_kwargs,
)

parser = argparse.ArgumentParser(description="Ask a question to the DB.")
parser.add_argument("question", type=str, help="The question to ask the DB")
args = parser.parse_args()

result = chain(
    {"question": args.question},
    return_only_outputs=True,
)
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
