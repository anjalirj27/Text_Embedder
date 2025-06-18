import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch

# Load public model
model_name = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Inference function
def get_embedding(text):
    # E5 models expect: "query: your text here"
    encoded_input = tokenizer("query: " + text, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = model_output.last_hidden_state[:, 0]  # CLS token
    normed = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normed[0].tolist()  # return list

# Gradio UI
iface = gr.Interface(fn=get_embedding,
                     inputs=gr.Textbox(label="Enter text"),
                     outputs=gr.Textbox(label="Embedding"),
                     title="Text Embedder")

iface.launch()
