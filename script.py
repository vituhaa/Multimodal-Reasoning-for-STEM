import streamlit as st
import torch
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

DEVICE = "cpu"

def render_formula(formula):
    formula = formula.strip()
    formula = re.sub(r'\s+', ' ', formula)
    formula = formula.replace('{ ', '{').replace(' }', '}')
    formula = re.sub(r'\^\s+', '^', formula)
    if not formula.startswith('$'):
        formula = f"${formula}$"
        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()

    t = ax.text(0.5, 0.5, formula,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='black',
        usetex=False)
    
    ax.figure.canvas.draw()
    bbox = t.get_window_extent()
    fig.set_size_inches(bbox.width/80,bbox.height/80)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


model_path = "vituuha/smolvlm-fine-tuned"
# model_path = "vituuha/smolvlm-full-add"
peft_config = PeftConfig.from_pretrained(model_path)
base_model_id = peft_config.base_model_name_or_path

processor = AutoProcessor.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    use_fast=True
)

processor.tokenizer.padding_side = "left"
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    trust_remote_code=True,
    low_cpu_mem_usage=False
).to(DEVICE)

model = PeftModel.from_pretrained(
    base_model,
    model_path,
    torch_dtype=torch.float32,
    is_trainable=False
)

model.eval()

st.title("SmolVLM-256M Instruct finetuning")
st.write("Load your image to convert hand-written mathematical formula into LaTex format.")

input_file = st.file_uploader("Choose image: ", type=["jpg", "jpeg", "png"])
if input_file is not None:
    image = Image.open(input_file).convert("RGB")
    st.image(image, caption="Your input image: ", width=600)
    with st.spinner("Thinking..."):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this hand-written mathematical formula into LaTex format."}
                ]
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        input_len = inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_len:]
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        answer = generated_texts[0].strip()

        st.success("Inference result: ")
        st.write(answer)

        output_img = render_formula(answer)
        st.image(output_img)
