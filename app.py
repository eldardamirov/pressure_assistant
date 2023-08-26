import gradio as gr

from datetime import datetime
from uuid import uuid4
from huggingface_hub import snapshot_download

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
from llama_cpp import Llama

import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader


SYSTEM_PROMPT = "Ты — русскоязычный автоматический ассистент МФЦ Санкт-Петербурга. Ты разговариваешь с людьми и помогаешь им. Отвечай лаконично и точно, используя только правду. Если необходимо - добавляй детали, списки документов."

SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


model_name = "ggml-model-q8_0.bin"
embedder_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

model = Llama(
    model_path=model_name,
    n_ctx=2000,
    n_parts=1,
    n_gpu_layers=12,
    n_threads=20,
)

max_new_tokens = 1500
embeddings = HuggingFaceEmbeddings(model_name=embedder_name)

def get_uuid():
    return str(uuid4())

def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def upload_files(files, file_paths):
    file_paths = [f.name for f in files]
    return file_paths


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text


def build_index(file_path, chunk_size, chunk_overlap):
    # loader = CSVLoader(file_path='/home/edamirov/Development/SPB_hack/langchain/data/train_dataset_Датасет.csv')
    # data = loader.load()
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(data)
    fixed_documents = []
    for doc in documents:
        doc.page_content = process_text(doc.page_content)
        if not doc.page_content:
            continue
        fixed_documents.append(doc)

    db = Chroma.from_documents(
        fixed_documents,
        embeddings,
        client_settings=Settings(
            anonymized_telemetry=False
        )
    )

    return db


def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history

db_ref = build_index('data/train_dataset_Датасет.csv', 250, 30)
retriever_ref = db_ref.as_retriever(search_kwargs={"k": 4})

db_parsed_data = build_index('data/mfc_context_chunks.csv', 500, 30)
retriever_parsed_data = db_parsed_data.as_retriever(search_kwargs={"k": 4})

def retrieve(history, retrieved_docs):
    context = ""
    
    print(f'HISTORY: {history}\n==================================\n')
    
    last_user_message = history[-1][0]
    # retriever = db.as_retriever(search_kwargs={"k": k_documents})
    docs_ref = retriever_ref.get_relevant_documents(last_user_message)
    retrieved_docs_ref = "\n\n".join([doc.page_content for doc in docs_ref])
    
    docs_parsed = retriever_parsed_data.get_relevant_documents(last_user_message)
    retrieved_docs_parsed = "\n\n".join([doc.page_content for doc in docs_parsed])
    
    retrieved_docs_buf = []
    if len(docs_ref) > 0:
        retrieved_docs_buf.append(docs_ref[0].page_content)
    if len(docs_parsed) > 0:
        retrieved_docs_buf.append(docs_parsed[0].page_content)
        
    for cur_doc in docs_parsed[1:]:
        retrieved_docs_buf.append(cur_doc.page_content)
    for cur_doc in docs_ref[1:]:
        retrieved_docs_buf.append(cur_doc.page_content)
    
        
    print(f'RETRIEVED DOCS, REFERENCE: {retrieved_docs_ref} \n')
    print('\n======================================\n')
    print(f'RETRIEVED DOCS, PARSED DATA: {retrieved_docs_parsed} \n')
    
    retrieved_docs = "\n\n".join(retrieved_docs_buf)
    
    return retrieved_docs


def dump_history_negative(history):
    current_datetime = datetime.now()

    buf = []
    cur_idx = 0
    for user_qry, assistant_ans in history:
        buf.append({'index': cur_idx, 'role': 'user', 'content': user_qry})
        cur_idx +=1
        buf.append({'index': cur_idx, 'role': 'assistant', 'content': assistant_ans})
        cur_idx +=1
    
    history_dump = pd.DataFrame(buf)
    history_dump.to_excel(f'./storage/negative_feedback/{current_datetime.strftime("%H_%M_%S#%d-%m-%Y")}.xlsx')
    
    return history

def dump_history_positive(history):
    current_datetime = datetime.now()

    buf = []
    cur_idx = 0
    for user_qry, assistant_ans in history:
        buf.append({'index': cur_idx, 'role': 'user', 'content': user_qry})
        cur_idx +=1
        buf.append({'index': cur_idx, 'role': 'assistant', 'content': assistant_ans})
        cur_idx +=1
    
    history_dump = pd.DataFrame(buf)
    history_dump.to_excel(f'./storage/positive_feedback/{current_datetime.strftime("%H_%M_%S#%d-%m-%Y")}.xlsx')
    
    return history



def bot(
    history,
    system_prompt,
    conversation_id,
    retrieved_docs
):
    if not history:
        return

    top_p = 0.9
    top_k = 30
    temp = 0.13
    
    tokens = get_system_tokens(model)[:]
    tokens.append(LINEBREAK_TOKEN)

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    if retrieved_docs:
        last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {last_user_message}"
    message_tokens = get_message_tokens(model=model, role="user", content=last_user_message)
    tokens.extend(message_tokens)

    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens.extend(role_tokens)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temp
    )

    partial_text = ""
    for i, token in enumerate(generator):
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
        yield history
    
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}"
) as demo:

    conversation_id = gr.State(get_uuid)

   
    chatbot = gr.Chatbot(label="Диалог").style(height=400)
   
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Отправить сообщение",
                placeholder="Отправить сообщение",
                # lines=2,
                max_lines=2,
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Отправить")
                stop = gr.Button("Остановить")
                clear = gr.Button("Новый диалог")
                
    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Отличный ответ", interactive=True)
        downvote_btn = gr.Button(value="👎  Плохой ответ", interactive=True)
        
    with gr.Row():
        retrieved_docs = gr.Textbox(
            lines=6,
            max_lines=8,
            label="Релевантные отсылки",
            placeholder="Задайте вопрос",
            interactive=False
        )


    # Pressing Enter
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, retrieved_docs],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            conversation_id,
            retrieved_docs
        ],
        outputs=chatbot,
        queue=True,
    )

    # Pressing the button
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, retrieved_docs],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            conversation_id,
            retrieved_docs
        ],
        outputs=chatbot,
        queue=True,
    )

    # Stop generation
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )

    # Clear history
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Negative vote
    upvote_btn.click(
        fn=dump_history_negative,
        inputs=[chatbot],
        outputs=None,
        queue=False
    )
    
    # Positive vote
    upvote_btn.click(
        fn=dump_history_positive,
        inputs=[chatbot],
        outputs=None,
        queue=False
    )

demo.queue(max_size=128, concurrency_count=1)
# demo.launch()
demo.launch(share=True)
