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
from giskard import Dataset, Model, testing, Suite, SuiteInput, slicing_function, SlicingFunction

# Set the OpenAI API Key environment variable.
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

# Display options.
pd.set_option("display.max_colwidth", None)

# Replace this with the path to your PDF file in the healthcare domain
HEALTHCARE_PDF_PATH = "Reports\AI-Powered Healthcare Report.pdf"

LLM_NAME = "gpt-3.5-turbo"

TEXT_COLUMN_NAME = "query"

PROMPT_TEMPLATE = """You are the Healthcare Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on the Healthcare system and also to address questions related to
patient illness history.
You will be given a question and relevant excerpts from the AI-Powered Healthcare Report .
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

def get_context_storage() -> FAISS:
    """Initialize a vector storage of embedded healthcare report chunks (context)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    docs = PyPDFLoader(HEALTHCARE_PDF_PATH).load_and_split(text_splitter)
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return db

# Create the chain.
llm = OpenAI(temperature=0.5)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
healthcare_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=get_context_storage().as_retriever(), prompt=prompt)

# Define a custom Giskard model wrapper for the serialization.
class FAISSHealthcareQA(Model):
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
healthcare_model = FAISSHealthcareQA(
    model=healthcare_qa_chain,
    model_type="text_generation",
    name="Healthcare Domain Question Answering",
    description="This model answers any question about the healthcare domain",
    feature_names=[TEXT_COLUMN_NAME]
)

# Optional: Wrap a dataframe of sample input prompts to validate the model wrapping and to narrow specific tests' queries.
healthcare_dataset = Dataset(pd.DataFrame({
    TEXT_COLUMN_NAME: [
        "Why is a homeopathic remedy listed as the primary treatment for type 2 diabetes? Is there strong scientific evidence for its effectiveness?",
        "What are the common side effects of chemotherapy?",
        "How does the healthcare system address mental health issues?",
    ]
}))

# Define your custom slicing function based on your requirements
# ...

# Define your custom slicing function based on your requirements
@slicing_function(name="Emotion sentiment", row_level=False, tags=["sentiment", "text"])
def emotion_sentiment_analysis(x: pd.DataFrame, column_name: str = "Why is a homeopathic remedy listed as the primary treatment for type 2 diabetes? Is there strong scientific evidence for its effectiveness?", emotion: str = "positive", threshold: float = 0.9) -> pd.DataFrame:
    """
    Slicing logic to filter rows with positive sentiment.
    """
    return x[column_name].apply(lambda text: emotion in text)


# Validate the wrapped model and dataset.
print(healthcare_model.predict(healthcare_dataset).prediction)

# Create a SuiteInput for your custom slicing function
shared_input_custom = SuiteInput("custom_slice", SlicingFunction)

# Create a Suite and add tests based on your evaluation metrics and slicing function
suite = (
    Suite()
    .add_test(testing.test_auc(slicing_function=shared_input_custom, threshold=0.9, column_name="Why is a homeopathic remedy listed as the primary treatment for type 2 diabetes? Is there strong scientific evidence for its effectiveness?", emotion="Slicing logic to filter rows with positive sentiment."))
    # Add more tests as needed for your specific evaluation metrics
)

# Run the suite with your healthcare text generation model and dataset
suite.run(model=healthcare_model, dataset=healthcare_dataset, custom_slice=emotion_sentiment_analysis, column_name="Why is a homeopathic remedy listed as the primary treatment for type 2 diabetes? Is there strong scientific evidence for its effectiveness?", emotion="Slicing logic to filter rows with positive sentiment.")

suite.run(
    model=healthcare_model,
    dataset=healthcare_dataset,
    custom_slice=emotion_sentiment_analysis,
    column_name="Why is a homeopathic remedy listed as the primary treatment for type 2 diabetes? Is there strong scientific evidence for its effectiveness?",  # Use the actual column name you want to test
    emotion="positive"  # Replace "positive" with the desired sentiment for testing
)
