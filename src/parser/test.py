from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

loader = AsyncChromiumLoader(['https://www.sravni.ru/karty/'])
html = loader.load()

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "li", "div", "a"]
)

print(docs_transformed[0].page_content[0:50000])