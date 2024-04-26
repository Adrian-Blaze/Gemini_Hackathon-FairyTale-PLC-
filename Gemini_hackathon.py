import os
import pandas as pd
from langchain.document_loaders import CSVLoader
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import tempfile
from langchain_experimental.agents import create_csv_agent


google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)

logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpJNvC_FVVBEFDwD-HMmwFmE1IIUVQbxZ7Vg&s"
logo_url2= 'data:image/jpeg;base64,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'

# Define HTML and CSS with placeholders
html_temp = f"""
<div style="display: flex; align-items: center; margin-bottom: 10px;">
  <img src="{logo_url2}" style="width: 40px; height: 40px; margin-right: 10px;" alt="Logo" />
  <h1 style="font-weight: bold; font-size: 30px; margin: 0;">Fairy Tale plc</h1>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)

#st.title("Fairy Tale plc")
st.markdown("# ")
st.subheader("Chat auto data query")
col1, col2 = st.columns(2)
# Add content to columns
with col1:
    uploaded_stock_records = st.file_uploader("Upload your inventory", type=["csv"])
        
    if uploaded_stock_records is not None:
        st.success('Stock record upload success', icon=None)

        loader = CSVLoader(uploaded_stock_records)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_stock_records.read())
            file_path = temp_file.name
        csv1 = file_path
with col2:
    uploaded_sales_records = st.file_uploader("Upload your sales record", type=["csv"])
            
    if uploaded_sales_records is not None:
        st.success('Sales record upload success', icon=None)
        loader = CSVLoader(uploaded_sales_records)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_sales_records.read())
            file_path = temp_file.name
        csv2 = file_path

llm =  ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", streaming=True,
                             temperature=0, google_api_key=google_api_key)

prompt = '''You are a smart store inventory manager. You go to market for the store to restock. At the close of business, the sales team also submit their record to you. 
            Any product with less thank 5 cartons left after sales need to be restocked(ignore all negative figures).
            You are to look through both records and compare to answer this in clear, well-structured markdown format while always limiting feedbaclk to 20 records-'''

prompt2 = ''' Smart Store Inventory Management Guidelines

- If you are greeted, respond politely and also prompt user to enter a query related to their records.

- As the inventory manager for a smart store, you play a crucial role in ensuring efficient restocking and accurate sales analysis. Here are the key guidelines for your weekly inventory management:

- Restocking Procedure: You are responsible for restocking the store once a week. Only products that have been depleted below a certain threshold after sales are to be restocked.

- Depleted Product Definition: A product is considered depleted if the quantity remaining after sales is less than 10. Use this criterion to determine which products require restocking.

- Sales Record Review: At the end of each business week, the sales team submits their records to you. Review these records carefully to identify any discrepancies or anomalies.

- Clarification Protocol: Always seek clarification if you encounter queries or instructions that are unclear or ambiguous. Open communication is essential for accurate inventory management.

- Resource Availability: If you find that you lack the necessary tools or information to respond accurately to a query, promptly communicate this and seek assistance from relevant stakeholders.

- Response Methodology: All responses to queries must be derived from deductive reasoning based on a thorough examination of the uploaded documents. Clearly articulate any deductions made and justify your conclusions.
- Highlight corresponding feedback in clear, well-structured markdown format.
These guidelines are designed to ensure consistency, accuracy, and transparency in inventory management practices. Adherence to these principles will contribute to the overall success of the smart store operation.
'''
st.markdown("#")
st.markdown("#")
query = st.text_input("Ask anything about your records!")
# Without streaming response
#ask_button = st.button("Ask")
#if ask_button:
# Perform prediction or any desired action
  #  with st.spinner("Query in progress..."):
 #       agent = create_csv_agent(llm, [csv1, csv2], verbose=True)

#        response = agent.run(prompt+query)
        

 #       styled_container = '''
  #      <div style="background-color: #F5F5F5; padding: 10px; border-radius: 5px;">
   #         <p style="font-size: 20px; color: gray; font-weight: bold;">{}</p>
    #    </div>
     #   '''.format(response)
#
#        st.write(styled_container, unsafe_allow_html=True)
ask_button = st.button("Ask")
if ask_button:
    # Perform prediction or any desired action
    with st.spinner("Query in progress..."):
        agent = create_csv_agent(llm, [csv1, csv2], verbose=True)

        # Start streaming the response
        full_response = ""
        response_placeholder = st.empty()  # Placeholder for displaying the response
        response_placeholder.markdown(full_response + "▌")


        # Generate the response in chunks and stream it
        for chunk in agent.run(prompt + query):
            full_response += chunk

            # Update the response placeholder with the latest content
            response_placeholder.write(full_response, unsafe_allow_html=True)


