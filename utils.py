import re
import os
import requests
import json
import math
from dotenv import load_dotenv, dotenv_values
load_dotenv()


def clean_text(text):

    # Remove extra new lines
    text = re.sub(r'\n+', '\n', text)

    # Remove extra whitespaces between words
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove consecutive numbers on new lines
    text = re.sub(r'\n\d+\n\d+(\n\d+)*', '\n', text)

    return text


def generate_embeddings(texts, max_list_length=128):
    try:
        voyageai_api_key = os.getenv("VOYAGE_API_KEY")
        voyageai_api_url = "https://api.voyageai.com/v1/embeddings"

        if not voyageai_api_key:
            raise Exception(
                "Please set the environment variable 'VOYAGE_API_KEY' to a valid VoyageAI API key.")

        embeddings = []
        batches_num = math.ceil(len(texts) / max_list_length)

        for i in range(0, len(texts), max_list_length):
            input_texts = texts[i:i + max_list_length]
            response = requests.post(
                voyageai_api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {voyageai_api_key}"
                },
                data=json.dumps(
                    {"input": input_texts, "model": "voyage-large-2"})
            )
            data = response.json()
            input_embeddings = [item['embedding'] for item in data['data']]
            embeddings.extend(input_embeddings)

        return embeddings
    except Exception as err:
        print("[error@get_voyage_embeddings]: ", err)
        return {"error": str(err)}


def upsert_to_pinecone(vectors, max_list_length=100):
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_api_url = "https://kb2-cen1257.svc.aped-4627-b74a.pinecone.io/vectors/upsert"
        # batches_num = math.ceil(len(vectors) / max_list_length)

        for i in range(0, len(vectors), max_list_length):
            input_vectors = vectors[i:i + max_list_length]
            response = requests.post(
                pinecone_api_url,
                headers={
                    "Content-Type": "application/json",
                    "Api-Key": pinecone_api_key
                },
                data=json.dumps({"vectors": input_vectors})
            )
            data = response.json()

        return True

    except Exception as err:
        print("[error@upsert_to_pinecone]: ", err)
        return False


def embed_and_upsert(texts):
    embeddings = generate_embeddings(texts)
    if "error" in embeddings:
        print("[error@embed_and_upsert]: Failed to get embeddings")
        return False

    vectors = [
        {
            "id": str(index),
            "values": embedding,
            "metadata": {"text": text_chunk, "purpose": "test"},
        }
        for index, (text_chunk, embedding) in enumerate(zip(texts, embeddings))
    ]

    success = upsert_to_pinecone(vectors)
    return success


def reset_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_api_url = "https://kb2-cen1257.svc.aped-4627-b74a.pinecone.io/vectors/delete"
    response = requests.post(
        pinecone_api_url,
        headers={
            "Content-Type": "application/json",
            "Api-Key": pinecone_api_key
        },
        data=json.dumps({
            "deleteAll": True
        })
    )
    data = response.json()
    return data


def get_relevant_chunks(text, top_k=10, only_text=False):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_api_url = "https://kb2-cen1257.svc.aped-4627-b74a.pinecone.io/query"
    embeddings = generate_embeddings([text])

    response = requests.post(
        pinecone_api_url,
        headers={
            "Content-Type": "application/json",
            "Api-Key": pinecone_api_key
        },
        data=json.dumps({
            "vector": embeddings[0],
            "topK": top_k,
            "includeMetadata": True,
        })
    )
    data = response.json()

    if ("error" in data):
        print("[error@get_relevant_chunks]: ", data["error"])

    if (only_text):
        data["matches"] = [item["metadata"]["text"]
                           for item in data["matches"]]

    return data["matches"]


def rerank_chunks(query, chunks):
    voyageai_api_key = os.getenv("VOYAGE_API_KEY")
    voyageai_api_url = "https://api.voyageai.com/v1/rerank"

    response = requests.post(
        voyageai_api_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {voyageai_api_key}"
        },
        data=json.dumps({
            "query": query,
            "documents": chunks,
            "model": "rerank-1"
        })
    )

    jsonRes = response.json()
    data = jsonRes["data"]
    # Add the associated text chunk with the reranked score
    data = [{"score": item["relevance_score"], "index": item["index"], "id": item["index"],
             "text": chunks[item["index"]]} for item in data]

    return data


def print_chunks(chunks, more_info=False):
    print("Total Chunks: ", len(chunks))

    for i, c in enumerate(chunks):
        print(f"--------\nCHUNK {i+1}\n--------")
        if (more_info):
            print(f"Id: {c.get('id', 'N/A')}, Score: {c.get('score', 'N/A')}")
            print(c.get("text") or c.get("metadata", {}).get("text", "N/A"))
        else:
            print(c)
