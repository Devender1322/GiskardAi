import os
from pathlib import Path

import openai
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, load_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from IPython.display import  display
from giskard import Dataset, Model, scan, GiskardClient

# OPENAI_API_KEY = "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"

# Set the OpenAI API Key environment variable.
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

# Display options.
pd.set_option("display.max_colwidth", None)

from langchain.chat_models import ChatOpenAI



IPCC_REPORT_URL = "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf"

LLM_NAME = "gpt-3.5-turbo"

TEXT_COLUMN_NAME = "query"

PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

def get_context_storage() -> FAISS:
    """Initialize a vector storage of embedded IPCC report chunks (context)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    docs = PyPDFLoader(IPCC_REPORT_URL).load_and_split(text_splitter)
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return db


# Create the chain.
llm = OpenAI(temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=get_context_storage().as_retriever(), prompt=prompt)

# Test the chain.
climate_qa_chain("Is sea level rise avoidable? When will it stop?")

# Define a custom Giskard model wrapper for the serialization.
class FAISSRAGModel(Model):
    def model_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[TEXT_COLUMN_NAME].apply(lambda x: self.model.run({"query": x}))

    def save_model(self, path: str):
        out_dest = Path(path)
        # Save the chain object
        self.model.save(out_dest.joinpath("model.json"))

        # Save the FAISS-based retriever
        db = self.model.retriever.vectorstore
        db.save_local(out_dest.joinpath("faiss"))

    @classmethod
    def load_model(cls, path: str) -> Chain:
        src = Path(path)

        # Load the FAISS-based retriever
        db = FAISS.load_local(src.joinpath("faiss"), OpenAIEmbeddings())

        # Load the chain, passing the retriever
        chain = load_chain(src.joinpath("model.json"), retriever=db.as_retriever())
        return chain


# Wrap the QA chain
giskard_model = FAISSRAGModel(
    model=climate_qa_chain,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type="text_generation",  # Either regression, classification or text_generation.
    name="Climate Change Question Answering",  # Optional.
    description="This model answers any question about climate change based on IPCC reports",  # Is used to generate prompts during the scan.
    feature_names=[TEXT_COLUMN_NAME]  # Default: all columns of your dataset.
)

# Optional: Wrap a dataframe of sample input prompts to validate the model wrapping and to narrow specific tests' queries.
giskard_dataset = Dataset(pd.DataFrame({
    TEXT_COLUMN_NAME: [
        "According to the IPCC report, what are key risks in the Europe?",
        "Is sea level rise avoidable? When will it stop?"
    ]
}))

# Validate the wrapped model and dataset.
print(giskard_model.predict(giskard_dataset).prediction)

# results = scan(giskard_model, giskard_dataset, only="hallucination")

# display(results)

# test_suite = results.generate_test_suite("Test suite generated by scan")
# test_suite.run()

full_results = scan(giskard_model, giskard_dataset)

display(full_results)
full_results.to_html("LLM_reports.html")
test_suite = full_results.generate_test_suite("Test suite generated by scan")
test_suite.run()