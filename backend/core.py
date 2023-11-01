import os
# import sys
from dotenv import load_dotenv
import pinecone
from typing import Any, List, Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain

from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from consts import INDEX_NAME

load_dotenv()

# template_message = [
#     SystemMessage(
#         content="You are a health care consultant and your role is to answer customer inquiries."
#     )
# ]

system_message_prompt = PromptTemplate.from_template(
    template="""You are a health care consultant and your role is to answer customer inquiries in Korean language."""  
)

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)

def run_llm(query: str, chat_history: List[dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(client=None)

    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-3.5-turbo")
    # chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4")

    _systemtemplate = """    
    """
    
    _template = """
    You are a consultant supporting small business owners at a small business distribution center in Korea, 
    and your role is to answer customer inquiries in Korean.
    Given the following conversation and follow-up questions:
    Turn follow-up questions into stand-alone questions.
    Add 'Answer the question' at the end of the standalone question. If you don't know the answer, say 'I'm sorry' in Korean.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        # system_prompt=system_prompt,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )

    return qa({"question": query, "chat_history": chat_history})

if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
