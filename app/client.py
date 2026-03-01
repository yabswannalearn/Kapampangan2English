import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/translate"

st.title("Kapampangan Translator")
st.write("Powered by a fine tuned Qwen 1.7B and RAG")

query = st.text_input("Enter a Kapampangan word or sentence:")

if st.button("Translate"):
    if query.strip():
        with st.spinner("Translating..."):
            try:
                # post request
                response = requests.post(API_URL, json={"query": query})
                response.raise_for_status()

                # extract and display result
                result = response.json()
                st.success(result["translations"])

            except requests.exceptions.ConnectionError:
                st.error("Failed to connect.")
            except Exception as e:
                st.error(f"An error occured: {e}")
    else:
        st.warning("Please enter text to translate")