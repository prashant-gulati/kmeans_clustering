import gradio as gr
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def extract_colors(input_img, k):
    if input_img is None:
        return None, None, "Please upload an image."
    
    # Convert PIL image to NumPy array
    img_arr = np.array(input_img)
    
    # If the image has an alpha channel to control opacity/transparency (RGBA), convert to RGB
    if img_arr.shape[-1] == 4:
        img_arr = img_arr[..., :3]
    
    # Reshape the image to a 2D array of pixels (N_pixels, N_channels)
    # Each row is a pixel, and columns are R, G, B values
    original_shape = img_arr.shape
    pixels = img_arr.reshape(-1, 3)
    
    #n_init specifies the number of times the k-means algorithm will be run with different centroid seeds.
    model = KMeans(n_clusters=int(k), n_init=10, random_state=42)
    model.fit(pixels)
    
    # Array of shape (k, 3). Each row represents one of the 'palette' colors found by the algorithm.
    colors = model.cluster_centers_.astype(int)
    # The labels for each pixel (which cluster it belongs to)
    labels = model.labels_
    
    # Fancy indexing: For every pixel's label, it looks up the corresponding RGB value in colors. The result is a long list of RGB pixels representing the entire image.
    # Reshape it back into the original height, width, and color channels of your image so it can be displayed as a picture.
    segmented_image = colors[labels].reshape(original_shape)

    # Build the color palette
    palette = np.zeros((100, k * 100, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        palette[:, i*100:(i+1)*100] = color
    
    return segmented_image, palette, f"Extracted {k} colors: {colors.tolist()}"

with gr.Blocks(title="🎨 K-means clustering - Extract dominant colors") as demo:
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

demo.launch()