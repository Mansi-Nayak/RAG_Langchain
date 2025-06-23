"""
This script performs question answering over a YouTube video's Hindi 
transcript.It fetches the transcript using `youtube_transcript_api`.
Then it splits the transcript into chunks for better processing.
Each chunk is embedded using HuggingFace's Sentence Transformers.
The embeddings are stored and searched using a FAISS vector store.
A retriever fetches relevant chunks given a user query.
A lightweight causal LLM (Falcon-RW) is used to generate text on CPU.
The prompt restricts the LLM to only answer based on transcript content.
LangChain's composable pipeline is used to manage this flow.
Finally, the script asks a sample question and prints the model's answer.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

# Step 1: Fetch transcript from YouTube
video_id = "J5_-l7WIO_w"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    transcript = ""
    print("No captions available for this video.")

# Step 2: Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 3: Convert text to embeddings using Sentence Transformers
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embedding_model)

# Step 4: Setup retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Step 5: Load a small causal model for CPU (e.g., TinyLlama, Falcon-RW, etc.)
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    device=-1,  # CPU only
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Step 6: Prompt template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)


# Step 7: Combine into a runnable pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# Step 8: Ask question
question = "Can you summarize the video?"
answer = main_chain.invoke(question)

print(answer)
