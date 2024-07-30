import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


# Streamlit app
st.title("Chat with CSV files")

st.write("Upload a CSV file, enter your OpenAI API key, and select a model to chat with your data using GPT models.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# API key input
api_key = st.text_input("Enter your OpenAI API key", type="password")

# Model selection
model_options = ["gpt-3.5-turbo-0125","gpt-4-turbo","gpt-4o"]
selected_model = st.selectbox("Select a GPT model", model_options)

if uploaded_file and api_key:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Initialize the language model
        llm = ChatOpenAI(temperature=0, model=selected_model, api_key=api_key)

        # Create the dataframe agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True  # Add this line to handle parsing errors
        )

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Chat message input
        user_question = st.chat_input("Ask a question about the data:")
        prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
            There can only be two types of chart, "bar" and "line".
            
            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}
            
            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}
            
            Return all output as a string.
            
            All strings in "columns" list and data list, should be in double quotes,
            
            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}
            
            Lets think step by step.
            
            Below is the query.
            Query: 
            """
        + user_question
    )


        
        if user_question:
            try:
                # Get the response from the agent
                response = agent.invoke(prompt)
                st.session_state.chat_history.append((prompt, response))
            except Exception as e:
                st.error(f"Error running query: {e}")

        # Display chat history
                # Display chat history
        for question, answer in st.session_state.chat_history:
           ## st.chat_message("user").text(question)
            st.chat_message("assistant").json(answer)

    except Exception as e:
        st.error(f"Error initializing the model or processing the file: {e}")
else:
    st.info("Please upload a CSV file, enter your OpenAI API key, and select a model to proceed.")
