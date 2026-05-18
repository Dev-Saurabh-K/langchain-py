from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression", metadata={'source':'H1'}),
    Document(page_content="drink vodka and live in hospital walays with doctors", metadata={'source':'H1'}),
    Document(page_content="sex is great for health", metadata={'source':'H1'}),
    Document(page_content="Tomorrow is our viva", metadata={'source':'H1'}),
    Document(page_content="Your girlfriend is so sexy!", metadata={'source':'H1'}),
    Document(page_content="Gym keeps you healthy", metadata={'source':'I4'}),
    Document(page_content="eating apple a day keeps the doctor away", metadata={'source':'I5'})

]


embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# print(vectorstore)


# similarity_retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

similarity_retriver = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5})

query = 'what is good for health?'


results = similarity_retriver.invoke(query)

for result in results:
    print(result.page_content)
    print('\\n')

