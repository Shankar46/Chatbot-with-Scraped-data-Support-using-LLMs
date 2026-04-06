import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------ CONFIG ------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

requests.packages.urllib3.disable_warnings()


# ------------------ CLEAN TEXT ------------------
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    return text


# ------------------ REMOVE HTML NOISE ------------------
def extract_main_content(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ")


# ------------------ SCRAPE ------------------
def scrape_html(url):
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        all_text = []
        main_text = clean_text(extract_main_content(soup))
        all_text.append(main_text)

        links = list(set([
            link.get("href") for link in soup.find_all("a", href=True)
            if is_valid_link(link.get("href"))
        ]))

        # 🔥 PRIORITY LINKS
        priority_keywords = ["chairman", "about", "board", "who"]
        priority_links = [
            link for link in links
            if any(keyword in link.lower() for keyword in priority_keywords)
        ]

        st.write(f"Scraping {min(10, len(priority_links))} important links...")

        for link in priority_links[:10]:
            try:
                sub_res = requests.get(link, verify=False)
                sub_res.raise_for_status()
                sub_soup = BeautifulSoup(sub_res.content, 'html.parser')

                sub_text = clean_text(extract_main_content(sub_soup))
                all_text.append(sub_text)
            except:
                continue

        process_chunks(all_text)

    except Exception as e:
        st.error(f"Error: {e}")


# ------------------ CHUNKING ------------------
def process_chunks(text_list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []
    for text in text_list:
        chunks.extend(splitter.split_text(text))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index")

    st.success("Data processed successfully!")


# ------------------ MODEL ------------------
def get_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt_template = """
You are an intelligent assistant.

Strictly answer the question using ONLY the context below.

Rules:
- Give a SHORT and DIRECT answer
- Do NOT include extra information
- If the answer is not found, say: "Not available in context"

Context:
{context}

Question:
{question}

Answer (one line only):
"""

    return model, prompt_template


# ------------------ QUERY ------------------
def user_input(query):
    if not os.path.exists("faiss_index"):
        st.error("Please scrape data first!")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.max_marginal_relevance_search(query, k=8)

    model, prompt_template = get_model()

    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt_template.format(context=context, question=query)

    with st.spinner("Thinking..."):
        response = model.invoke(final_prompt)

    st.markdown(f"### 🤖 Answer:\n{response.content}")

    if docs:
        st.write("📌 Top Source:", docs[0].page_content[:300])


# ------------------ FILTER LINKS ------------------
def is_valid_link(link):
    if not link:
        return False
    return not re.search(r'\.(pdf|jpg|png|docx?)$', link, re.I)


# ------------------ UI ------------------
def main():
    st.set_page_config(page_title="Web RAG")
    st.header("🌐 Chat with Scraped Website (Improved RAG)")

    url = st.text_input("Enter Website URL")

    if st.button("Scrape"):
        if url:
            scrape_html(url)
        else:
            st.warning("Please enter a URL")

    query = st.text_input("Ask Question")

    if st.button("Ask"):
        if query:
            user_input(query)
        else:
            st.warning("Please enter a question")


if __name__ == "__main__":
    main()
