import os 
from dotenv import load_dotenv 
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.7, 
    google_api_key = os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are helpful assistant. Answer this: {question}"
)

#output parser 
output_parser = StrOutputParser()

#chain using LCEL 
qa_chain = prompt | llm | output_parser


#Streamlit UI
st.set_page_config(page_title="Gemini Q&A Bot", page_icon="🤖")
st.title("Gemini Q&A bot")

if "history" not in st.session_state:
    st.session_state.history = []


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything", key="input")
    submitted = st.form_submit_button("Send")

if user_input:
    response = qa_chain.invoke({"question": user_input})
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Gemini", response))

for sender, message in st.session_state.history:
    if sender == "You":
        st.markdown(f"**🧑‍💻 You:** {message}")
    else:
        st.markdown(f"**🤖 Gemini:** {message}")