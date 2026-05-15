# for static webpages 


# for synamic or javascript heavy , use seleniumloader

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text -\n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()


url = 'https://www.amazon.in/Samsung-Storage-MediaTek-Charging-Upgrades/dp/B0FN7QTRPY/ref=sr_1_2?_encoding=UTF8&content-id=amzn1.sym.e16f4584-b1af-493d-b12c-13fce2efddbe&dib=eyJ2IjoiMSJ9.4f1rX_5_Vo-IE9pM8FtjNVnXwneHDv3be4KQCydN1fSEDAQfKOTrXgeKFbyngD1q88XZcMF-UBUohXNIPtIA6JCj92a86odBKni6adCOeQUquwJH_U1B-dYfCRcesAtahxKpmuUthSU9msYfLD-EUc8-lFMrIz_YIiok24gDZ5OpxSVx7o4RdfTriZhfFZofSG5x6OM2CHZXxkLJiEztv9rxhWh6kWw5f6o0_Kv7F3sWdUK0m49HLJ5IwqbzWOUWZMKAvilXotnUufTCnHPsVRHelYiaOX4cWoA_wCZltjc.DZqTJMeZspa0KXUkVGYsehTR99ZIU-EseN9yzBc_GKE&dib_tag=se&pd_rd_r=6c8ba0e3-d6f4-46dd-827b-63e7f8e74106&pd_rd_w=kr0Q7&pd_rd_wg=DdqXf&qid=1778840296&refinements=p_36%3A1318505031&rnid=1318502031&s=electronics&sr=1-2'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the processor used in the product', 'text':docs[0].page_content}))
# print(len(docs))

# print(docs[0].page_content)

