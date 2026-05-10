from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv();

embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=300)

documents = [
    "Virat Kohli is The modern-day run machine known for passion, consistency, and chasing greatness.",
    "MS Dhoni is Captain Cool who mastered pressure and led India to historic victories.",
    "Sachin Tendulkar is The God of Cricket who inspired generations with his legendary batting.",
    "Rohit Sharma is The Hitman famous for effortless sixes and record-breaking double centuries.",
    "Jasprit Bumrah is India’s pace spearhead known for deadly yorkers and unmatched accuracy."
    ]

query="tell me about bumrah"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings=embedding.embed_query(query)

scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

# print(scores)
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is: ", score)