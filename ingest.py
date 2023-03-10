#%%
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

persist_directory = "db_spells"
with open("data/dnd_spell_split.txt") as f:
    dnd_spell = f.read()
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
)
texts = text_splitter.split_text(dnd_spell)

docs = text_splitter.create_documents([dnd_spell])
embeddings = OpenAIEmbeddings()

metadatas = []
for i in texts:
    source = i.split("\n")[0]
    metadatas.append({"source": f"Spell {source} in dnd_spell_split.txt"})
#%%
docsearch = Chroma.from_texts(
    texts,
    embeddings,
    persist_directory=persist_directory,
    metadatas=metadatas,
)
docsearch.persist()
docsearch = None

# %%
