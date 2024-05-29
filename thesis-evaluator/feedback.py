from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from mistralai import exceptions
import os
import utils as ut
import httpx

load_dotenv()
ut.configure_logging()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
chat = ChatMistralAI(model = "open-mixtral-8x7b", mistral_api_key=MISTRAL_API_KEY)
embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
logger = ut.get_logger()

#Loading documents
def load_docs(path):
    loaders = [PyPDFLoader(os.path.join(path, file)) for file in os.listdir(path)]
    pages = []
    for loader in loaders:
        pages += loader.load()
    logger.info("Successfully loaded documents.")
    return pages

#Creating vector database for RAG
def create_vector_database(pages):
    #Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 6500,
        chunk_overlap = 0,
        length_function = len,
    )
    docs = text_splitter.split_documents(pages)
    #Create and save FAISS vector store
    logger.info("Creating embeddings...")
    library = FAISS.from_documents(docs, embeddings)
    library.save_local("faiss_index")
    logger.info("Successfully created embeddings.")
    return

#Creating retrieval chain for RAG
def create_chain(k):
    #Prompt for chat history
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    #Prompt for RAG retriever
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])
    #Loading FAISS vector store and initialising retriever
    faiss_saved = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_saved.as_retriever(search_kwargs={"k": k})
    #Chaining document and chat history chains for RAG
    doc_chain = create_stuff_documents_chain(chat, history_prompt)
    history_retriever_chain = create_history_aware_retriever(chat, retriever, retriever_prompt)
    retrieval_chain = create_retrieval_chain(history_retriever_chain, doc_chain)
    logger.info("Successfully created retrieval chain.")
    return retrieval_chain

#Getting feedback for input thesis
def get_feedback(name, retrieval_chain, check):
    chat_history = [SystemMessage(content="You are an evaluator of academic theses.")]
    #Query to check if document submitted is a thesis
    check_query = f'''
When given a name of thesis, analyse the actual thesis and make sure that it is an academic paper.

# Thesis:
{name}

# Instructions:
## Analyse the thesis:
Analyse the thesis provided. If something other than an academic paper was provided, do NOT say anything else, and say ONLY the following:
This document does not appear to be a thesis. Please make sure that you submitted the correct paper.
'''
    #Main query to evaluate thesis
    main_query = f'''
When given a name of thesis, analyse the actual thesis and review the questions that follow. Afterward, write a summarisation in the specified format that answers all the questions.

# Thesis:
{name}

# Instructions:
## Questions:
Abstract:
 - Does the abstract give enough vital information about the thesis?
 - Does the abstract briefly describe the main features of the work done?
Introduction:
 - Does the thesis introduce the project well for non-technical readers?
 - Does the thesis introduce the aims and goals of the project as well as the approach used?
 - Does the thesis present an overview of its contents?
Background:
 - Does the thesis provide the reader with important information to understand the thesis?
 - Does the thesis discuss the target audience of the system and the anticipated benefits that it brings?
 - Does the thesis explain the concepts used well with adequate references?
Literature Review:
 - Does the thesis discuss previous studies in the area of the thesis while highlighting the strengths and weaknesses of the study?
 - Does the thesis perform a critical analysis of other studies?
 - Does the thesis review state-of-the-art material in the designated area?
 - Does the thesis justify why the literature chosen to review is relevant to the project?
Specification and Design:
 - Does the thesis give the reader a clear picture of the system/project created?
 - Does the thesis discuss and justify the design choices made in the thesis?
Methodology or Implementation:
 - Does the thesis describe the system used in fine detail?
 - Does the thesis describe any problems that may have arisen during the development of the system/project and how they were solved?
 - Does the thesis avoid describing large chunks of code?
 - Does the thesis perform a critical analysis on the operation of the system/project?
Testing and Evaluation:
 - Does the thesis describe the methods used to evaluate the working state of the system?
 - Does the thesis use evidence from the literature about similar systems to justify why those specific tests were used to evaluate the system?
 - Does the thesis provide a demonstration that the system works, or doesn't work, as intended?
 - Does the thesis include comprehensible summaries of the results of all the critical tests that have been made?
 - Does the thesis describe the strengths and weaknesses of the system based on the test results?
 - Does the thesis perform a comparison of practical with theoretical results and their interpretation?
 - Does the thesis perform a comparison with published work?
Future Work and Conclusions:
 - Does the thesis describe any ideas for future work?
 - Does the thesis summarise the contents of the projects as well as the main results?

## Summarise your answer:
Answer ALL of the previous questions in a well-written, formal and comprehensive summary. Explain the reasoning for your answers. Do NOT invent any information.
Write your summarisation in the following format while speaking CRITICALLY:
Abstract: 
\n
[Summary of Abstract]
\n
Introduction:
\n
[Summary of Introduction Answers]
\n
Background:
\n
[Summary of Background Answers]
\n
Literature Review:
\n
[Summary of Literature Review]
\n
Specification and Design:
\n
[Summary of Specification and Design]
\n
Methodology or Implementation:
\n
[Summary of Methodology or Implementation]
\n
Evaluation and Testing:
\n
[Summary of Evaluation and Testing]
\n
Future Work and Conclusions:
\n
[Summary of Future Work and Conclusions]
\n
'''
    try:
        logger.info("Getting feedback...")
        #If document has not been confirmed as thesis check it
        if not check:
            response = retrieval_chain.invoke({
                "input": check_query,
                "chat_history": chat_history
            })
            #If document is thesis perform evaluation
            if "This document does not appear to be a thesis. Please make sure that you submitted the correct paper." not in response["answer"]:
                check = True
                response = retrieval_chain.invoke({
                    "input": main_query,
                    "chat_history": chat_history
                    })
        else:
            response = retrieval_chain.invoke({
                        "input": main_query,
                        "chat_history": chat_history
                        })
        return check, response["answer"]
    except exceptions.MistralAPIException:
        logger.error(f"Error. Too many tokens in query. Max token limit is 32k.")
        return check, "The evaluator failed to process your document. Your text may be too long. Please make sure all appendices have been removed and re-upload it."
    except httpx.HTTPStatusError:
        logger.error(f"Error. Too many tokens in query. Max token limit is 32k.")
        return check, "The evaluator failed to process your document. Your text may be too long. Please make sure all appendices have been removed and re-upload it."
    except Exception as e:
        logger.error(f"Unexpected error occurred. Error: {str(e)}")
        return check, "An unexpected error occurred."
    

#Loading input thesis
# thesis = load_docs('Input/')

#Taking name of thesis for prompt later
# name = thesis[0]

#Creating vector database with guidelines and input thesis
# pages = thesis
# create_vector_database(pages)

#Creating retrieval chain and getting feedback from model
# retrieval_chain = create_chain()
# get_feedback(name, retrieval_chain)

#Theses are recommended to be inputted without appendices due to context limit.