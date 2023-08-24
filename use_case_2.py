from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer


def scan_website():
    urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    return docs


if __name__ == '__main__':
    # read_documents_and_index()
    docs = scan_website()
    print("Docs=", docs)
    print(docs[0].page_content[1000:2000])

