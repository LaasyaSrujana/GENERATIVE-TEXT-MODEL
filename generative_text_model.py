# ------------------------------------------------------
#  GPT-2 Text Generator with Gradio Web App
# ------------------------------------------------------

#  STEP 1: Install Required Libraries
!pip install transformers gradio --quiet

# STEP 2: Import Libraries
from transformers import pipeline
import gradio as gr

#  STEP 3: Load Pre-trained GPT-2 Model
generator = pipeline("text-generation", model="gpt2")

#  STEP 4: Define Generation Function
def generate_text(prompt, length, temperature):
    output = generator(
        prompt,
        max_length=length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=50256  # prevent warnings
    )
    return output[0]['generated_text']

#  STEP 5: Build Gradio Interface
title = "📄 GPT-2 Text Generator"
description = "Enter a topic or prompt, and let GPT-2 generate coherent paragraphs. Adjust the length and creativity."

interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g. The future of AI in medicine is..."),
        gr.Slider(minimum=50, maximum=300, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature (Creativity)")
    ],
    outputs="text",
    title=title,
    description=description,
    theme="default"
)

#  STEP 6: Launch App with Public Link
interface.launch(share=True)  # Set share=True to generate a shareable link
