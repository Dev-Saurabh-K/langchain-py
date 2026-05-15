from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# for small docs and immedite load
# docs = loader.load()


# for large docs aqnd one by one load
docs = loader.lazy_load()

# print(len(docs))
for document in docs:
    print(document.metadata)