import os
import streamlit as st
from dotenv import load_dotenv
from pprint import pprint



load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# print(os.getenv("GROQ_API_KEY"))
# print(os.getenv("HF_TOKEN"))

from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# from langchain import PromptTemplate, GROQ_LLM, StrOutputParser, Memory, RunnablePassthrough


# CSV Import


from functools import lru_cache

# lru_cache(maxsize=None)
# def load_documents():
#     loader = DirectoryLoader("data2/", glob="**/*.txt")
#     docs_all = loader.load()
#     return docs_all

# @st.cache_resource
# def invoke_model(query):
#     return app.invoke(query)

def invoke_model(query):
    print(query)
    return app.invoke(query)

# docs_all = load_documents()
# print("======= Loaded documents =======")
# print(len(docs_all))


# from langchain.text_splitter import RecursiveCharacterTextSplitter

# #splitting the text into
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
# texts = text_splitter.split_documents(docs_all)

# # from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# model_name = "BAAI/bge-base-en"
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

# embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cuda'},
#     encode_kwargs=encode_kwargs
# )

# from langchain_chroma import Chroma

# persist_directory = 'db'

import main 


vectordb = main.vectordb
# vectordb = Chroma.from_documents(documents=texts,
#                                  embedding=embedding,
#                                  persist_directory=persist_directory)

# RAG
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

GROQ_LLM = ChatGroq(
            model="llama3-70b-8192",
        )


# Web Search Chain
wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
# print(web_search_tool.invoke("What is apple news"))


#RAG CHAIN
# rag_prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n

#      <|eot_id|><|start_header_id|>user<|end_header_id|>
#     QUESTION: {question} \n
#     CONTEXT: {context} \n
#     Answer:
#     <|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["question","context"],
# )

rag_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the provided context to answer the question accurately. If you don't know the answer, simply state that you don't know. Provide a concise answer in no more than three sentences.\n

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    QUESTION: {question} \n
    CONTEXT: {context} \n
    Answer:
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"],
)

rag_prompt_chain = rag_prompt | GROQ_LLM | StrOutputParser()

rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | rag_prompt
    | GROQ_LLM
    | StrOutputParser()
)


from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# from langchain_core.output_parsers import JsonOutputParser

def write_markdown_file(content, filename):
  """Writes the given content as a markdown file to the local directory.

  Args:
    content: The string content to write to the file.
    filename: The filename to save the file as.
  """
  if type(content) == dict:
    content = '\n'.join(f"{key}: {value}" for key, value in content.items())
  if type(content) == list:
    content = '\n'.join(content)
  with open(f"{filename}.md", "w") as f:
    f.write(content)




#Categorize answer
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Support Categorizer Agent for ExxonMobil, a global leader in motor-oils, energy, and chemical manufacturing. You are an expert at understanding customer inquiries and categorizing them into useful categories based on the ExxonMobil Thailand website. Keep in mind that lubricant and motor-oils are the same and these products are the main focus of the company.

     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the question provided and categorize it into one of the following categories:
        motor-oils - used when someone is asking about motor oils products and services \
        fuel - used when someone is asking about fuel products and services \
        retailers - used when someone is asking about retail services and locations \
        promotions - used when someone is asking about ongoing promotions and offers \
        support - used when someone is seeking support or has issues related to ExxonMobil services \
        corporate-information - used when someone is asking about corporate information and news\
        customer_feedback - used when someone is giving feedback about a product \
        previous_interaction - used when the question is related to a previous interaction with chatbot. keyword: previous of my question or answer. Other thing, tend to have sentiment like this sentence \
        other - used when the question does not fit into any of the above categories \

            Output a single category only from the types ('motor-oils', 'fuel', 'retailers', 'promotions', 'support', 'corporate-information', 'customer_feedback', 'previous_interaction', 'other') \
            eg:
            'motor-oils' \

    QUESTION CONTENT:\n\n{initial_question}\n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_question"],
)

question_category_generator = prompt | GROQ_LLM | StrOutputParser()


## Research Router

research_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at reading the initial question and routing it to our internal knowledge system or directly to a draft response. \n

    Use the following criteria to decide how to route the draft: \n\n

    If the initial question only requires a simple response
    Just choose 'draft_response' for questions you can easily answer, prompt engineering, and common queries.
    If the question is just saying thank you etc then choose 'draft_response'.

    If you are unsure or the person is asking a question you don't understand then choose 'research_info'

    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use 'research_info'.
    Give a binary choice 'research_info' or 'draft_response' based on the question. Return the a JSON with a single key 'router_decision' and
    no preamble or explanation. Use both the initial question and the question category to make your decision.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question to route INITIAL_QUESTION: {initial_question} \n
    QUESTION_CATEGORY: {question_category} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question","question_category"],
)
research_router = research_router_prompt | GROQ_LLM | JsonOutputParser()

search_rag_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at formulating the best questions to ask our knowledge agent to obtain the most accurate and helpful information for the customer.

    Given the INITIAL_QUESTION and QUESTION_CATEGORY, determine the best questions that will provide the most useful information for crafting the final response. Focus on accurate and helpful details related to ExxonMobil's products and services. Direct your questions to our knowledge system, not to the customer.
    
    If the INITIAL_QUESTION is comparing two products of ExxonMobil, determine for the benefits and the different of each product, then suggest the best product to the customer. However, if the question is about the benefits of a specific product, provide the benefits of that product and do not compare.

    Return a JSON with a single key 'questions' containing up to 3 strings, with no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_QUESTION: {initial_question} \n
    QUESTION_CATEGORY: {question_category} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question","question_category"],
)


question_rag_chain = search_rag_prompt | GROQ_LLM | JsonOutputParser()
research_info = None

draft_writer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Answer Draft Writer Agent for ExxonMobil. Take the INITIAL_QUESTION below from a human that has contacted us, the question_category assigned by the categorizer agent, and the research from the research agent, and write a helpful response in a thoughtful and friendly way, but do not greet.
    Furthermore, make sure to write easy to read and understand responses that are helpful to the customer. I want you to conclude the draft in short and answer in paragraph.
            If the question is 'other', ask for more details.
            If the question is 'customer_feedback', thank them and address their feedback.
            If the question is 'corporate-information', provide relevant corporate info.
            If the question is 'support', assure them we value their concern and address the issue.
            If the question is 'promotions', provide details on current promotions.
            If the question is 'retailers', provide information about retail services or locations.
            If the question is 'fuel', provide information about fuel products and services.
            If the question is 'motor-oils', provide information about motor oils products and services.

            Do not invent information that hasn't been provided by the research_info or in the initial_question.

            Return the response in a JSON with a single key 'answer_draft' and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_QUESTION: {initial_question} \n
    QUESTION_CATEGORY: {question_category} \n
    RESEARCH_INFO: {research_info} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question","question_category","research_info"],
)


draft_writer_chain = draft_writer_prompt | GROQ_LLM | JsonOutputParser()

question_category = 'customer_feedback'
research_info = None


rewrite_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert at evaluating draft responses for customers and deciding if they need to be rewritten to improve quality.

    Use the following criteria to decide if the DRAFT_ANSWER needs to be rewritten:

    If the INITIAL_QUESTION only requires a simple response which the DRAFT_ANSWER contains, then it doesn't need to be rewritten.
    If the DRAFT_ANSWER addresses all the concerns of the INITIAL_QUESTION, then it doesn't need to be rewritten.
    If the DRAFT_ANSWER is missing information that the INITIAL_QUESTION requires, then it needs to be rewritten.

    Give a binary choice 'rewrite' (for needs to be rewritten) or 'no_rewrite' (for doesn't need to be rewritten) based on the DRAFT_ANSWER and the criteria.
    Return a JSON with a single key 'router_decision' and no preamble or explanation.

    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_QUESTION: {initial_question} \n
    QUESTION_CATEGORY: {question_category} \n
    DRAFT_ANSWER: {draft_answer} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question","question_category","draft_answer"],
)


rewrite_router = rewrite_router_prompt | GROQ_LLM | JsonOutputParser()

draft_analysis_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
     You are the Quality Control Agent. Read the INITIAL_QUESTION below from a customer, the question_category that the categorizer agent assigned, and the research from the research agent, and write an analysis of the DRAFT_ANSWER.

    Check if the DRAFT_ANSWER addresses the customer's issues based on the question category and the content of the initial question.

    Provide feedback on how the draft can be improved and what specific things can be added or changed to make the draft more effective at addressing the customer's issues.

    Do not make up or add information that hasn't been provided by the research_info or in the initial_question.

    Return the analysis in a JSON with a single key 'draft_analysis' and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_QUESTION: {initial_question} \n\n
    QUESTION_CATEGORY: {question_category} \n\n
    RESEARCH_INFO: {research_info} \n\n
    DRAFT_ANSWER: {draft_answer} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question","question_category","research_info"],
)
draft_analysis_chain = draft_analysis_prompt | GROQ_LLM | JsonOutputParser()


question_category = 'motor-oil'
research_info = None
draft_answer = "Describe the benefit of Mobil Super Moto™ 20W-40"


rewrite_answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are the Final Answer Draft Agent. Read the draft analysis below from the QC Agent and use it to rewrite and improve the DRAFT_ANSWER to create a final answer.

    Do not make up or add information that hasn't been provided by the research_info or in the initial_question.

    Return the final Answer Draft as JSON with a single key 'final_answer' which is a string and no preamble or explanation.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    INITIAL_QUESTION: {initial_question} \n\n
    QUESTION_CATEGORY: {question_category} \n\n
    RESEARCH_INFO: {research_info} \n\n
    DRAFT_ANSWER: {draft_answer} \n\n
    DRAFT_ANSWER_FEEDBACK: {answer_analysis} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["initial_question",
                     "question_category",
                     "research_info",
                     "answer_analysis",
                     "draft_answer",
                     ],
)
rewrite_chain = rewrite_answer_prompt | GROQ_LLM | JsonOutputParser()

question_category = 'customer_feedback'
research_info = None
draft_answer = "Yo we can't help you, best regards Sarah"


from langchain.schema import Document
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_question: answer
        question_category: answer category
        draft_answer: LLM generation
        final_answer: LLM generation
        research_info: list of documents
        info_needed: whether to add search info
        num_steps: number of steps
    """
    initial_question : str
    question_category : str
    draft_answer : str
    final_answer : str
    research_info : List[str] 
    info_needed : bool
    num_steps : int
    draft_answer_feedback : dict
    rag_questions : List[str]

def categorize_answer(state):
    """take the initial answer and categorize it"""
    print("---CATEGORIZING INITIAL Question---")
    initial_question = state['initial_question']
    num_steps = int(state['num_steps'])
    num_steps += 1

    question_category = question_category_generator.invoke({"initial_question": initial_question})
    print(question_category)
    # if question_category == 'previous_interaction':
    #     if 'generated' not in st.session_state:
    #         st.session_state.get('past', [])
    #     if 'past' not in st.session_state:
    #         st.session_state.get('generated', [])
    #     st.write(st.session_state.get('past', []))
    #     st.write(st.session_state.get('generated', []))

        # return 
    
    return {"question_category": question_category, "num_steps":num_steps}

conclusion_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant tasked with concluding answers based on provided contexts that relate to QUESTION. Use the information from CONTEXT results to provide a final, comprehensive answer. If the information is insufficient, state that you don't know the answer.
    Furthermore, make sure to write easy to read and understand responses that are helpful to the customer in short. Answer in paragraph.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    QUESTION: {question} \n
    CONTEXT: {context} \n
    Answer:
    
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "context"],
)
conclusion_agent = conclusion_prompt | GROQ_LLM | StrOutputParser()


previous_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant tasked with concluding answers based on provided contexts that relate to QUESTION. Use the information from CONTEXT results to provide a final, comprehensive answer. Furthermore, you need to answer the QUESTION from previous prompt of user which is CONTEXT. If the information is insufficient, state that you don't know the answer.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    QUESTION: {question} \n
    CONTEXT: {context} \n
    Answer:
    
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "context"],
)
previous_prompt_agent = previous_prompt | GROQ_LLM | StrOutputParser()



def research_info_search(state):
    # print("---RESEARCH INFO RAG---")
    
    print("---RESEARCH INFO RAG---")
    initial_question = state["initial_question"]
    question_category = state["question_category"]  # Remove leading/trailing whitespace
    num_steps = state['num_steps']
    num_steps += 1
    

    questions = question_rag_chain.invoke({"initial_question": initial_question,
                                                "question_category": question_category })
    questions = questions['questions']
    print(questions)
    final_results = []
    for question in questions:
        print(question)
        temp_docs = rag_chain.invoke(question)
        print(temp_docs)
        web_docs = web_search_tool.invoke(question)
        print(web_docs)
        # Combine the contexts from both sources
        combined_context = temp_docs + "\n\n" + web_docs
        question_results = question + '\n\n' + combined_context + "\n\n\n"

        final_answer = conclusion_agent.invoke({"question": question, "context": question_results})
        final_results.append(final_answer)
    
    print("==== Web Search Result ====")
    print(web_docs)
    print("==== FiNal Result ====")
    print(final_results)
    return {"research_info": final_results, "rag_questions": questions, "num_steps": num_steps}

    # Web search
    

    

#     else:
#         questions = question_rag_chain.invoke({"initial_question": initial_question,
#                                                 "question_category": question_category })
#         questions = questions['questions']
#         print(questions)
#         rag_results = []
#         final_results = []
#         for question in questions:
#             print(question)
#             temp_docs = rag_chain.invoke(question)
#             print(temp_docs)
#             web_docs = web_search_tool.invoke(question)
#             print(web_docs)
#             # Combine the contexts from both sources
#             combined_context = temp_docs + "\n\n" + web_docs
#             question_results = question + '\n\n' + combined_context + "\n\n\n"

#             final_answer = conclusion_agent.invoke({"question": question, "context": question_results})
#             final_results.append(final_answer)
#             # if rag_results is not None:
#             #     rag_results.append(question_results)
#             # else:
#             #     rag_results = [question_results]
#         print("==== Web Search Result ====")
#         print(web_docs)
#         print("==== FiNal Result ====")
#         print(final_results)
#         print(type(rag_results))
#         # write_markdown_file(rag_results, "research_info")
#         # write_markdown_file(questions, "rag_questions")
#         return {"research_info": final_results,"rag_questions":questions, "num_steps":num_steps}
# # return {"research_info": rag_results,"rag_questions":questions, "num_steps":num_steps}

def draft_answer_writer(state):
    print("---DRAFT ANSWER WRITER---")
    ## Get the state
    initial_question = state["initial_question"]
    question_category = state["question_category"]
    research_info = state["research_info"]
    num_steps = state['num_steps']
    num_steps += 1

    # Generate draft answer
    draft_answer = draft_writer_chain.invoke({"initial_question": initial_question,
                                     "question_category": question_category,
                                     "research_info":research_info})
    print(draft_answer)
    # print(type(draft_answer))

    answer_draft = draft_answer['answer_draft']
    # write_markdown_file(answer_draft, "draft_answer")

    return {"draft_answer": answer_draft, "num_steps":num_steps}

def analyze_draft_answer(state):
    print("---DRAFT ANSWER ANALYZER---")
    ## Get the state
    initial_question = state["initial_question"]
    question_category = state["question_category"]
    draft_answer = state["draft_answer"]
    research_info = state["research_info"]
    num_steps = state['num_steps']
    num_steps += 1

    # Generate draft answer
    draft_answer_feedback = draft_analysis_chain.invoke({"initial_question": initial_question,
                                                "question_category": question_category,
                                                "research_info":research_info,
                                                "draft_answer":draft_answer}
                                               )
    # print(draft_answer)
    # print(type(draft_answer))

    # write_markdown_file(str(draft_answer_feedback), "draft_answer_feedback")
    return {"draft_answer_feedback": draft_answer_feedback, "num_steps":num_steps}

def rewrite_answer(state):
    print("---REWRITE DRAFT ---")
    ## Get the state
    initial_question = state["initial_question"]
    question_category = state["question_category"]
    draft_answer = state["draft_answer"]
    research_info = state["research_info"]
    draft_answer_feedback = state["draft_answer_feedback"]
    num_steps = state['num_steps']
    num_steps += 1

    # Generate draft answer
    final_answer = rewrite_chain.invoke({"initial_question": initial_question,
                                                "question_category": question_category,
                                                "research_info":research_info,
                                                "draft_answer":draft_answer,
                                                "answer_analysis": draft_answer_feedback}
                                               )

    # write_markdown_file(str(final_answer), "final_answer")
    return {"final_answer": final_answer['final_answer'], "num_steps":num_steps}



def no_rewrite(state):
    print("---NO REWRITE DRAFT ---")
    ## Get the state
    draft_answer = state["draft_answer"]
    num_steps = state['num_steps']
    num_steps += 1

    # write_markdown_file(str(draft_answer), "final_answer")
    return {"final_answer": draft_answer, "num_steps":num_steps}

def state_printer(state):
    """print the state"""
    print("---STATE PRINTER---")
    print(f"Initial answer: {state['initial_question']} \n" )
    print(f"answer Category: {state['question_category']} \n")
    print(f"Draft answer: {state['draft_answer']} \n" )
    print(f"Final answer: {state['final_answer']} \n" )
    print(f"Research Info: {state['research_info']} \n")
    print(f"RAG Questions: {state['rag_questions']} \n")
    print(f"Num Steps: {state['num_steps']} \n")
    return

def route_to_research(state):
    """
    Route answer to web search or not.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """

    print("---ROUTE TO RESEARCH---")
    initial_question = state["initial_question"]
    question_category = state["question_category"]


    router = research_router.invoke({"initial_question": initial_question,"question_category":question_category })
    print(router)
    # print(type(router))
    print(router['router_decision'])
    if router['router_decision'] == 'research_info':
        print("---ROUTE answer TO RESEARCH INFO---")
        return "research_info"
    elif router['router_decision'] == 'draft_answer':
        print("---ROUTE answer TO DRAFT answer---")
        return "draft_answer"
    
def route_to_rewrite(state):

    print("---ROUTE TO REWRITE---")
    initial_question = state["initial_question"]
    question_category = state["question_category"]
    draft_answer = state["draft_answer"]
    # research_info = state["research_info"]

    # draft_answer = "Yo we can't help you, best regards Sarah"

    router = rewrite_router.invoke({"initial_question": initial_question,
                                     "question_category":question_category,
                                     "draft_answer":draft_answer}
                                   )
    print(router)
    print(router['router_decision'])
    if router['router_decision'] == 'rewrite':
        print("---ROUTE TO ANALYSIS - REWRITE---")
        return "rewrite"
    elif router['router_decision'] == 'no_rewrite':
        print("---ROUTE answer TO FINAL answer---")
        return "no_rewrite"
    

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("categorize_answer", categorize_answer) # categorize answer
workflow.add_node("research_info_search", research_info_search) # web search
workflow.add_node("state_printer", state_printer)
workflow.add_node("draft_answer_writer", draft_answer_writer)
workflow.add_node("analyze_draft_answer", analyze_draft_answer)
workflow.add_node("rewrite_answer", rewrite_answer)
workflow.add_node("no_rewrite", no_rewrite)

workflow.set_entry_point("categorize_answer")

# workflow.add_conditional_edges(
#     "categorize_answer",
#     route_to_research,
#     {
#         "research_info": "research_info_search",
#         "draft_response": "draft_answer_writer",
#     },
# )

workflow.add_edge("categorize_answer", "research_info_search")
workflow.add_edge("research_info_search", "draft_answer_writer")


workflow.add_conditional_edges(
    "draft_answer_writer",
    route_to_rewrite,
    {
        "rewrite": "analyze_draft_answer",
        "no_rewrite": "no_rewrite",
    },
)
workflow.add_edge("analyze_draft_answer", "rewrite_answer")
workflow.add_edge("no_rewrite", "state_printer")
workflow.add_edge("rewrite_answer", "state_printer")
workflow.add_edge("state_printer", END)

# Compile
app = workflow.compile()


# def invoke_model(query):
#     return app.invoke(query)