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


similarity_retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})


# for mmr retriever
# similarity_retriver = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs = {"k": 5}),
    llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
)

query = 'How to improve energy level and maintain balance?'


similarity_result = similarity_retriver.invoke(query)
multiquery_result = multiquery_retriever.invoke(query)



# results = similarity_retriver.invoke(query)

# for result in results:
#     print(result.page_content)
#     print('\\n')

for i in range(0,5):
    print(similarity_result[i].page_content)
    print(multiquery_result[i].page_content)


# multiquery give more accurate and relevant answer
