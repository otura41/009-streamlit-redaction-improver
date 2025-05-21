# import streamlit as st
# from langchain_openai import OpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.evaluation.qa import QAEvalChain

# def generate_response(
#     uploaded_file,
#     openai_api_key,
#     query_text,
#     response_text
# ):
#     #format uploaded file
#     documents = [uploaded_file.read().decode()]
    
#     #break it in small chunks
#     text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=0
#     )
#     texts = text_splitter.create_documents(documents)
#     embeddings = OpenAIEmbeddings(
#         openai_api_key=openai_api_key
#     )
    
#     # create a vectorstore and store there the texts
#     db = FAISS.from_documents(texts, embeddings)
    
#     # create a retriever interface
#     retriever = db.as_retriever()
    
#     # create a real QA dictionary
#     real_qa = [
#         {
#             "question": query_text,
#             "answer": response_text
#         }
#     ]
    
#     # regular QA chain
#     qachain = RetrievalQA.from_chain_type(
#         llm=OpenAI(openai_api_key=openai_api_key),
#         chain_type="stuff",
#         retriever=retriever,
#         input_key="question"
#     )
    
#     # predictions
#     predictions = qachain.apply(real_qa)
    
#     # create an eval chain
#     eval_chain = QAEvalChain.from_llm(
#         llm=OpenAI(openai_api_key=openai_api_key)
#     )
#     # have it grade itself
#     graded_outputs = eval_chain.evaluate(
#         real_qa,
#         predictions,
#         question_key="question",
#         prediction_key="result",
#         answer_key="answer"
#     )
    
#     response = {
#         "predictions": predictions,
#         "graded_outputs": graded_outputs
#     }
    
#     return response

# st.set_page_config(
#     page_title="Evaluate a RAG App"
# )
# st.title("Evaluate a RAG App")

# with st.expander("Evaluate the quality of a RAG APP"):
#     st.write("""
#         To evaluate the quality of a RAG app, we will
#         ask it questions for which we already know the
#         real answers.
        
#         That way we can see if the app is producing
#         the right answers or if it is hallucinating.
#     """)

# uploaded_file = st.file_uploader(
#     "Upload a .txt document",
#     type="txt"
# )

# query_text = st.text_input(
#     "Enter a question you have already fact-checked:",
#     placeholder="Write your question here",
#     disabled=not uploaded_file
# )

# response_text = st.text_input(
#     "Enter the real answer to the question:",
#     placeholder="Write the confirmed answer here",
#     disabled=not uploaded_file
# )

# result = []
# with st.form(
#     "myform",
#     clear_on_submit=True
# ):
#     openai_api_key = st.text_input(
#         "OpenAI API Key:",
#         type="password",
#         disabled=not (uploaded_file and query_text)
#     )
#     submitted = st.form_submit_button(
#         "Submit",
#         disabled=not (uploaded_file and query_text)
#     )
#     if submitted and openai_api_key.startswith("sk-"):
#         with st.spinner(
#             "Wait, please. I am working on it..."
#             ):
#             response = generate_response(
#                 uploaded_file,
#                 openai_api_key,
#                 query_text,
#                 response_text
#             )
#             result.append(response)
#             del openai_api_key
            
# if len(result):
#     st.write("Question")
#     st.info(response["predictions"][0]["question"])
#     st.write("Real answer")
#     st.info(response["predictions"][0]["answer"])
#     st.write("Answer provided by the AI App")
#     st.info(response["predictions"][0]["result"])
#     st.write("Therefore, the AI App answer was")
#     st.info(response["graded_outputs"][0]["results"])

import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

def generate_response(
    uploaded_file,
    openai_api_key,
    query_text,
    response_text
):
    # Formatear archivo subido
    documents = [uploaded_file.read().decode()]
    
    # Dividir en fragmentos pequeños
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )
    
    # Crear vectorstore y almacenar los textos
    db = FAISS.from_documents(texts, embeddings)
    
    # Crear interfaz de recuperación
    retriever = db.as_retriever()
    
    # Diccionario de QA real
    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]
    
    # Cadena QA regular
    qachain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )
    
    # Predicciones
    predictions = qachain.apply(real_qa)
    
    # Cadena de evaluación
    eval_chain = QAEvalChain.from_llm(
        llm=OpenAI(openai_api_key=openai_api_key)
    )
    # Autoevaluación
    graded_outputs = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }
    
    return response

# Configuración de la página
st.set_page_config(
    page_title="Evaluador de RAG App",
    page_icon="📊",
    layout="centered"
)
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>📊 Evaluador de RAG App</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ ¿Cómo funciona?"):
    st.markdown("""
    <div style='color: #444; font-size: 16px;'>
    Sube un documento, haz una pregunta y compara la respuesta generada por IA con la respuesta real.<br>
    Así puedes evaluar si la IA responde correctamente o está alucinando.
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📄 Sube un documento .txt",
    type="txt"
)

if not uploaded_file:
    st.info("Por favor, sube un archivo .txt para comenzar.")

query_text = st.text_input(
    "❓ Escribe una pregunta que ya hayas verificado:",
    placeholder="Escribe aquí tu pregunta",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "✅ Escribe la respuesta real a la pregunta:",
    placeholder="Escribe aquí la respuesta confirmada",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = st.text_input(
        "🔑 Clave API de OpenAI:",
        type="password",
        disabled=not (uploaded_file and query_text and response_text)
    )
    submitted = st.form_submit_button(
        "Evaluar",
        disabled=not (uploaded_file and query_text and response_text)
    )
    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.error("Por favor, introduce una clave API de OpenAI válida.")
        else:
            with st.spinner("⏳ Procesando..."):
                try:
                    response = generate_response(
                        uploaded_file,
                        openai_api_key,
                        query_text,
                        response_text
                    )
                    result.append(response)
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
            del openai_api_key

if len(result):
    response = result[-1]
    st.markdown("---")
    st.subheader("Resultados de la evaluación")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pregunta:**")
        st.info(response["predictions"][0]["question"])
        st.markdown("**Respuesta real:**")
        st.success(response["predictions"][0]["answer"])
    with col2:
        st.markdown("**Respuesta de la IA:**")
        st.warning(response["predictions"][0]["result"])
        st.markdown("**Evaluación:**")
        st.info(response["graded_outputs"][0]["results"])

