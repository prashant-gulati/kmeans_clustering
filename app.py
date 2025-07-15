#import gradio as gr

#def greet(name):
#    return "Hello " + name + "!!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")
#demo.launch()

import gradio as gr
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def extract_colors(input_img, k):
    if input_img is None:
        return None, None, "Please upload an image."
    img_arr = np.array(input_img)
    if img_arr.shape[-1] == 4:
        img_arr = img_arr[..., :3]
    original_shape = img_arr.shape
    pixels = img_arr.reshape(-1, 3)
    model = KMeans(n_clusters=int(k), n_init=10, random_state=42)
    model.fit(pixels)
    colors = model.cluster_centers_.astype(int)
    labels = model.labels_
    segmented_image = colors[labels].reshape(original_shape)
    palette = np.zeros((100, k * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        palette[:, i*100:(i+1)*100] = color
    return segmented_image, palette, f"Extracted {k} colors: {colors.tolist()}"

css = ".narrow { max-width: 350px !important; }"

with gr.Blocks(css=css, title="🎨 K-means clustering - Extract dominant colors") as demo:
    gr.Markdown("# 🎨 K-means clustering - Extract dominant colors")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Image")
            k_radio = gr.Radio(choices=[2, 3, 4, 5, 6], value=4, label="Number of Colors (k)")
            submit_btn = gr.Button("Submit", variant="primary")

            example_container = gr.Column()

        with gr.Column():
            out_seg = gr.Image(label="Image w dominant colors")
            out_pal = gr.Image(label="Colors extracted")
            out_text = gr.Textbox(label="RGB values")

    with example_container:
        gr.Examples(
            examples=[["artist.jpeg"], ["cat.jpg"], ["quote.jpeg"], ["skull.jpg"], ["woman.jpeg"]],
            inputs=[input_img],
            label=None,
        )

    submit_btn.click(fn=extract_colors, inputs=[input_img, k_radio], outputs=[out_seg, out_pal, out_text])

demo.launch(share=True)