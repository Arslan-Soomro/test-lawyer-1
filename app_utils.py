from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from utils import get_relevant_chunks, rerank_chunks, print_chunks


def ask_legal_assistant(query, chat_history):
    model = ChatAnthropic(model="claude-3-sonnet-20240229")
    pinecone_namespace = "simple-rag-2"

    # Repharse query as a standalone question
    repharse_query_template = [
        ("system",
         "Given the following conversation, repharase the followup question to be a standalone question, that is understandable and optimized for similarity search while preserving the meaning. The output should just be the rephrased question \n Chat History: \n {chat_history}"),
        ("human", "followup: \n {question}")
    ]
    rephrase_query_prompt = ChatPromptTemplate.from_messages(
        repharse_query_template)
    chat_history_string = "\n".join(
        [f'{d["role"]}: {d["content"]}' for d in chat_history])
    rephrase_query_res = model.invoke(rephrase_query_prompt.format(
        question=query, chat_history=chat_history_string))
    rephrased_query = rephrase_query_res.content
    print("Rephrased Query: ", rephrased_query);
    # Get the relevant chunks of text
    relevant_chunks = get_relevant_chunks(query, pinecone_namespace, 10)
    relevant_chunks_text = [item["metadata"]["text"]
                            for item in relevant_chunks]
    rerank_results = rerank_chunks(query, relevant_chunks_text)
    most_relevant_chunks = rerank_results[0:3]
    relevant_chunks_text = "\n----------------------\n".join(
        [c["text"] for c in most_relevant_chunks])
    # print_chunks(most_relevant_chunks, True)

    # Get the Actual Answer from LLM
    get_answer_template = [
        ("system", "You are a legal assistant, you are given some knowledge about the topic, and a question. You must answer user's question only if its evident from the knowledge, If you cannot find or extract enough knowledge, let the user know that as per your limited knowledge you are not sure and won't be able to answer them"),
        ("human",
         "Knowledge is made up of top text chunks in order, acquired after performing a similarity search in the knowledge base for user's query \n Knowledge: \n {relevant_chunks} \n question: {question}")
    ]
    get_answer_prompt = ChatPromptTemplate.from_messages(get_answer_template)
    print("Prompt: \n\n", get_answer_prompt.format(
        question=rephrased_query,
        relevant_chunks=relevant_chunks_text
    ))
    get_answer_res = model.invoke(get_answer_prompt.format(
        question=rephrased_query,
        relevant_chunks=relevant_chunks_text
    ))

    print("Answer: \n\n", get_answer_res)
    return get_answer_res.content

"""
ask_legal_assistant("what challenges are faced by them", [{
    "role": "human",
    "content": "Describe the hierarchy and organizational structure of the judicial system in Pakistan."
},
    {
    "role": "ai",
    "content": "The judicial system in Pakistan follows a hierarchical structure with the Supreme Court at the top, followed by High Courts in each province, and subordinate courts which include district and sessions courts, civil and judicial magistrates."
},
    {
    "role": "human",
    "content": "what is the role of subordinate courts"
},
    {
    "role": "ai",
    "content": "Subordinate courts handle the bulk of civil and criminal cases at the district level."
}
])
"""
