import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
def scrape_html(url):
    try:
        with requests.get(url, verify=False) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            scraped_chunks = []
            text_content = soup.get_text(strip=True)
            cleaned_text = clean_text(text_content) 
            scraped_chunks.append(cleaned_text)       
            # Find and scrape links within the page
            links = soup.find_all('a', href=True)
            ul=[]
            dl=[]
            if links:
                for link in links:
                    link_url = link.get('href')
                    if is_valid_link(link_url) and link_url not in ul:
                        ul.append(link_url)
                        
                    else:
                        dl.append(link_url)
                num_links = min(10, len(ul))  # Calculate the number of links to scrape
                st.write(f"Scraping {num_links} links from the UL list")
                for i, link_url in enumerate(ul[:num_links]):  # Limit the iteration to num_links
                    # Scrape all links from the ul list without limiting them
                    scraped = scrape_linked_page(link_url)
                    if scraped:
                        scraped_chunks.append(scraped) 
            st.write(ul) 
            st.write("Duplicate links are give below")
            st.write(dl)                      
            print_and_process_chunks(scraped_chunks)
    except requests.RequestException as e:
        st.error(f"Error: Could not retrieve HTML content from {url}. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping {url}. Error: {e}")
def scrape_linked_page(link_url):
    try:
        with requests.get(link_url, verify=False) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(strip=True)
            cleaned_text = clean_text(text_content)
            st.write("\nLinked Page Text content:")
            return cleaned_text
            # Optionally, you can perform further processing on the scraped text
    except requests.RequestException as e:
        st.error(f"Error: Could not retrieve HTML content from {link_url}. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping {link_url}. Error: {e}")
def clean_text(text):
    # Keep only English language characters, numbers, and common punctuation marks
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s.,]', '', cleaned_text)
    return cleaned_text

def print_and_process_chunks(text):
    all_chunks=[]
    for content in text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
    process_chunks(all_chunks)
def process_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.from_texts(chunks, embeddings)
    faiss_db.save_local("faiss_index")
def get_conversation_chain():
    prompt_template = """ 
Answer the question as detailed as possible based on the text content scraped from the web.\n
If the answer is not available in the scraped content, please indicate so.\n\n

Context:\n
{context}\n\n

Question:\n
{question}\n\n

Answer:

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
# response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])
        
def is_valid_link(link_url):
    # Check if the link URL is not a document, image, Gmail link, or contact number
    if (not re.match(r'.+\.(pdf|docx?|xlsx?|pptx?|jpg|jpeg|png|gif)', link_url, re.IGNORECASE) and 
        not 'mail.google.com' in link_url and 
        not re.match(r'tel:\d+', link_url)):
        if not 'mailto:e-reg.vpt@gov.in' in link_url and not '#' in link_url:
            return True
    else:
        return False
# For demonstration purposes, consider all links as valid

def main():
    st.set_page_config("Scrapped Data")
    st.header("Chat with Scrapped Data using Gemini")
    user_question = st.text_input("Ask a Question:")
    if st.button("Enter") and user_question:
        user_input(user_question)
    target_url = st.text_input("Enter the website link")   
    if st.button("Scrape HTML") and target_url:
        scrape_html(target_url)

if __name__ == "__main__":
    main()
