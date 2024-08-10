import os
import gradio as gr
from huggingface_hub import login
from transformers import load_tool
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import spaces

#login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda")

@spaces.GPU(duration=40)
def DocChat(question, history):
    print(question)
    if question["files"]:
        image = question["files"][-1]["path"]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0])==tuple:
                image = hist[0][0]

    if image is None:
      gr.Error("You need to upload an image for LLaVA to work.")
        
    prompt=f"[INST] <image>\n{question['text']} [/INST]"
    image = Image.open(image).convert("RGB")
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=500)
    outputmsg = processor.decode(output[0], skip_special_tokens=True)
    
    generated_text_without_prompt = outputmsg[len(prompt)-5:]
    yield generated_text_without_prompt
    
demo = gr.ChatInterface(fn=DocChat, title="Image Chatbot", description="Chat with your images/documents with LLaVA NeXT.",
                        stop_btn="Stop Generation", multimodal=True)

if __name__ == "__main__":
    demo.launch(debug=True)
