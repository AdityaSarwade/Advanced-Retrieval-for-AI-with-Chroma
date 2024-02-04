import time
import torch
import chromadb
import gradio as gr
from pypdf import PdfReader
from helper_utils import word_wrap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

reader = PdfReader("microsoft_annual_report_2022.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]


character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
print(f"\nTotal character chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
print(f"\nTotal token chunks: {len(token_split_texts)}")

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft_annual_report_2022", embedding_function=embedding_function
)

ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# Bits and bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/StableBeluga-7B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=bnb_config,
)


def retrieve_documents(query, chroma_collection):
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results["documents"][0]
    return retrieved_documents


def generate_answer(query):
    retrieved_documents = retrieve_documents(query, chroma_collection)
    # get the start time
    st = time.time()
    information = "\n\n".join(retrieved_documents)

    system_prompt = "### System: You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report. You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information.\n\n"
    prompt = (
        system_prompt
        + f"### User:\nQuestion: {query}.\nInformation: {information}\n\n### Assistant:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output = model.generate(
        **inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=2048
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st

    return output, elapsed_time


def main():
    # Gradio Interface
    with gr.Blocks(
        title="Overview of embeddings-based retrieval",
        theme="dark",
        css="footer {visibility: hidden}",
    ) as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Group():
                        input_text = gr.Textbox(
                            label="Input Query for Microsoft Annual Report 2022",
                        )
                with gr.Row():
                    generate_button = gr.Button("Generate")
                with gr.Row():
                    with gr.Group():
                        time_taken = gr.Textbox(
                            label="Time taken for execution",
                            interactive=False,
                            autofocus=True,
                        )
            with gr.Column():
                with gr.Row():
                    with gr.Group():
                        output_text = gr.Textbox(label="Output from model")
        generate_button.click(
            fn=generate_answer, inputs=[input_text], outputs=[output_text, time_taken]
        )

        demo.launch(debug=True, share=True)


main()
