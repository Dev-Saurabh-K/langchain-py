from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv();

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0.3)


st.header("Research Tool")

user_input=st.text_input("Enter your prompt")

if st.button("Summerize"):
    result=model.invoke(user_input)
    st.text(result.content)