import os
# import sys
# from dotenv import load_dotenv
from typing import Any, List, Tuple

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate

# from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
import pinecone

# sys.path.insert(0, "/home/hanson/workspace/ai/langchain/documentation-helper")

# load_dotenv()

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)

def run_llm(query: str, chat_history: List[dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(client=None)

    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4") #gpt-3.5-turbo

    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )
    
    # summary_template = """
    #      given the Linkedin information {information} about a person from I want you to create:
    #      1. a short summary
    #      2. two interesting facts about them
    #  """
     
    # summary_prompt_template = PromptTemplate(
    #     input_variables=["information"], template=summary_template
    # )

    _template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question. 
    At the end of standalone question add this 'Answer the question in Korean language.' If you do not know the answer reply with 'I am sorry' in Korean.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
