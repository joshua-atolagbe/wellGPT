class Langchain_RAG:
    def __init__(self,pdf_file_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.pdf_file_path = pdf_file_path
        print("loading pdf file , this may take time to process")
        self.loader = loader = PDFMinerLoader(self.pdf_file_path)
        self.data = self.loader.load()
        print("<< pdf file loaded")
        print("<< chunking")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
        self.texts = text_splitter.split_documents(self.data)
        print("<< chunked")
        self.get_vec_value= FAISS.from_documents(self.texts, self.embeddings)
        print("<< saved")
        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4})

    def __call__(self,query):
        rev = self.retriever.get_relevant_documents(query)
        return "".join([i.page_content for i in rev])