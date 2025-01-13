import pandas as pd
import json
import os
import logging
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import constants
import streamlit as st

# Load custom data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Create a LangChain application
def create_app(custom_data, memory=None):
    """Create a LangChain application with memory."""
    # Create an OpenAI model
    llm = ChatOpenAI(model="gpt-4o", api_key=constants.APIKEY)

    # Create a prompt template
    prompt_template = PromptTemplate(
        template=(
            "Given the conversation history and provided data, answer the following question: {input}\n"
            "Conversation History: {chat_history}\nData: {custom_data}\n"
            "Provide a concise answer based on both."
        ),
        input_variables=["input", "custom_data", "chat_history"]
    )

    # Create a conversation memory if not provided
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

    # Create an LLM chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory
    )

    return chain, memory

# Process data using the Agent and LLM model
def process_data(chain, question, custom_data):
    """Process data using the LangChain LLM model."""
    try:
        # Retrieve chat history from memory
        chat_history = chain.memory.buffer if chain.memory else ""
        print(f"Chat History: {chat_history}")

        # Run the chain with question, custom data, and chat history
        output = chain.invoke({
            "input": question,
            "custom_data": json.dumps(custom_data),
            "chat_history": chat_history
        })

        # Extract and return the final answer
        final_answer = output.get('text', '').split('Data:')[0].strip()
        return final_answer

    except Exception as e:
        return f"An error occurred while processing your question: {e}"

# Streamlit app
def main():
    st.title("💬 Chat with ESB")
    st.caption("🚀 An ESB chatbot powered by Amine and Elaa")

    # Initialize session state for chatbot messages and memory
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]
    if "memory" not in st.session_state:
        st.session_state["memory"] = None

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input box for user to type their question
    if prompt := st.chat_input("Type your question here..."):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Load and preprocess custom data
        custom_data = load_data('data/esb.json')

        # Create a LangChain application with persistent memory
        chain, memory = create_app(custom_data, st.session_state["memory"])

        # Process the user's question
        final_answer = process_data(chain, prompt, custom_data)

        # Append assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.chat_message("assistant").write(final_answer)

        # Update the conversation memory in the session state
        st.session_state["memory"] = memory

def app():
    main()