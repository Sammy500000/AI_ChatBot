import streamlit as st
from query_data import query_database

# Streamlit application title
st.title("AI ChatBot")

# Input box for user query
query_text = st.text_input("Enter your query:")

if st.button("Search"):
    if query_text.strip() == "":
        st.error("Please enter a query.")
    else:
        response, sources = query_database(query_text)
        st.subheader("Response:")
        st.write(response)

        st.subheader("Sources:")
        if sources:
            for source in sources:
                st.write(f"- {source}")
        else:
            st.write("No sources available.")
