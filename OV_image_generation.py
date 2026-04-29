import gradio as gr
import openvino_genai as ov_genai
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
import time
import threading
import random
import datetime

BASE_MODELS_DIR = "./models"
OUTPUT_DIR = "./output/pict"
LORAS_DIR = "./loras"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORAS_DIR, exist_ok=True)

def get_model_list():
    if not os.path.exists(BASE_MODELS_DIR):
        return []
    return [d for d in os.listdir(BASE_MODELS_DIR) 
            if os.path.isdir(os.path.join(BASE_MODELS_DIR, d)) 
            and os.path.exists(os.path.join(BASE_MODELS_DIR, d, "model_index.json"))]

def get_lora_list():
    if not os.path.exists(LORAS_DIR):
        return ["None"]
    loras = [f for f in os.listdir(LORAS_DIR) if f.endswith(".safetensors")]
    return ["None"] + loras

current_pipe = None
current_model_name = ""
current_lora_name = ""
current_lora_weight = 0.0

def load_model(model_name, lora_name, lora_weight, device):
    global current_pipe, current_model_name, current_lora_name, current_lora_weight
    
    # モデルもLoRA設定も同じなら、VRAMに乗っているロード済みのモデルをそのまま返す
    if current_model_name == model_name and current_lora_name == lora_name and current_lora_weight == lora_weight and current_pipe is not None:
        return current_pipe
    
    model_path = os.path.join(BASE_MODELS_DIR, model_name)
    print(f"Loading model: {model_name} to {device} (LoRA: {lora_name} @ {lora_weight})...")
    
    # LoRAが選択されている場合はAdapterConfigを設定してパイプラインを作る
    if lora_name != "None":
        lora_path = os.path.join(LORAS_DIR, lora_name)
        adapter_config = ov_genai.AdapterConfig()
        adapter = ov_genai.Adapter(lora_path)
        adapter_config.add(adapter, float(lora_weight))
        current_pipe = ov_genai.Text2ImagePipeline(model_path, device, adapters=adapter_config)
    else:
        current_pipe = ov_genai.Text2ImagePipeline(model_path, device)
        
    current_model_name = model_name
    current_lora_name = lora_name
    current_lora_weight = lora_weight
    return current_pipe

def prepare_generation(current_seed, mode):
    current_seed = int(current_seed)
    if mode == "random":
        new_seed = random.randint(0, 2147483647) 
    elif mode == "increase":
        new_seed = current_seed + 1
    elif mode == "decrease":
        new_seed = max(0, current_seed - 1) 
    else:
        new_seed = current_seed 
    return new_seed, time.time()

def generate(model_name, lora_name, lora_weight, prompt, negative_prompt, width, height, steps, cfg_scale, base_seed, seed_mode, batch_count, device, progress=gr.Progress()):
    progress(0, desc="Loading Model to VRAM...") 
    # LoRAの情報も渡してロードする
    pipe = load_model(model_name, lora_name, lora_weight, device)
    
    state = {
        "batch_idx": 0,
        "batch_total": int(batch_count),
        "step": 0,
        "step_total": int(steps),
        "done": False,
        "saved_paths": []
    }

    def run_inference():
        current_seed = int(base_seed)
        
        for i in range(state["batch_total"]):
            state["batch_idx"] = i + 1
            state["step"] = 0
            
            if i > 0:
                if seed_mode == "random":
                    current_seed = random.randint(0, 2147483647)
                elif seed_mode == "decrease":
                    current_seed = max(0, current_seed - 1)
                else: 
                    current_seed += 1

            def progress_callback(step, num_steps, latent):
                state["step"] = step + 1
                state["step_total"] = num_steps
                return False

            print(f"Generating image {i+1}/{batch_count} - Seed: {current_seed}, CFG: {cfg_scale}")
            
            output = pipe.generate(
                prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=float(cfg_scale),
                rng_seed=current_seed,
                callback=progress_callback
            )
            
            result_img = Image.fromarray(np.array(output.data[0]))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = current_model_name.replace("/", "_").replace("\\", "_")
            filename = f"{timestamp}_{safe_model_name}_seed{current_seed}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = PngInfo()
            metadata.add_text("prompt", prompt)
            metadata.add_text("negative_prompt", negative_prompt)
            metadata.add_text("seed", str(current_seed))
            metadata.add_text("steps", str(steps))
            metadata.add_text("cfg_scale", str(cfg_scale))
            metadata.add_text("width", str(width))
            metadata.add_text("height", str(height))
            metadata.add_text("model", current_model_name)
            metadata.add_text("lora", lora_name)
            metadata.add_text("lora_weight", str(lora_weight))
            
            result_img.save(save_path, pnginfo=metadata)
            state["saved_paths"].append(save_path)
            
        state["done"] = True
        state["final_seed"] = current_seed

    thread = threading.Thread(target=run_inference)
    thread.start()

    while not state["done"]:
        if state["step_total"] > 0:
            overall_progress = ((state["batch_idx"] - 1) + (state["step"] / state["step_total"])) / state["batch_total"]
            desc = f"Generating Image {state['batch_idx']}/{state['batch_total']} | Step {state['step']} / {state['step_total']}"
            progress(overall_progress, desc=desc)
        time.sleep(0.1) 

    thread.join()
    progress(1.0, desc="All Done!")
    
    return state["saved_paths"], state["saved_paths"], state["final_seed"]


def load_settings_from_image(image):
    if image is None:
        return [gr.update()] * 11
    
    info = image.info
    if not info:
        print("No metadata found in this image.")
        return [gr.update()] * 11

    res = []
    res.append(gr.update(value=info.get("model")))
    res.append(gr.update(value=info.get("lora", "None")))
    res.append(gr.update(value=float(info.get("lora_weight", 1.0))))
    res.append(gr.update(value=info.get("prompt")))
    res.append(gr.update(value=info.get("negative_prompt")))
    res.append(gr.update(value=int(info.get("width", 1024))))
    res.append(gr.update(value=int(info.get("height", 1024))))
    res.append(gr.update(value=int(info.get("steps", 28))))
    res.append(gr.update(value=float(info.get("cfg_scale", 7.5)))) 
    res.append(gr.update(value=int(info.get("seed", 12345))))
    res.append(gr.update(value="fix"))
    
    print("Settings restored from image metadata.")
    return res

model_choices = get_model_list()
lora_choices = get_lora_list()

with gr.Blocks(title="Image generation with OpenVINO") as demo:
    gr.Markdown("# Image generation with OpenVINO")
    
    start_time_state = gr.State()
    save_path_state = gr.State()
    final_seed_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_drop = gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None, label="Select Model", scale=3)
                # LoRAのリストを最新に更新するボタン
                refresh_btn = gr.Button("🔄 Reload", scale=1, min_width=1) 
            
            # LoRAの選択肢と重みスライダー
            with gr.Row():
                lora_drop = gr.Dropdown(choices=lora_choices, value="None", label="Select LoRA (from ./loras/)")
                lora_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA Weight")
            
            device_radio = gr.Radio(["GPU", "CPU"], value="GPU", label="Device")
            
            prompt = gr.Textbox(label="Prompt", lines=4)
            neg_prompt = gr.Textbox(label="Negative Prompt", lines=2)
            
            with gr.Row():
                width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
            
            with gr.Row():
                steps = gr.Slider(1, 50, value=28, step=1, label="Steps", scale=1)
                cfg_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="CFG Scale", scale=1) 
            
            with gr.Row():
                seed_mode = gr.Radio(["fix", "increase", "decrease", "random"], value="fix", label="Seed Mode", scale=2)
                seed = gr.Number(value=12345, label="Seed", precision=0, scale=1)
        
        with gr.Column(scale=1):
            with gr.Row():
                batch_count = gr.Slider(1, 100, value=1, step=1, label="Batch Count (生成枚数)", scale=1)
                
                gen_btn = gr.Button("Generate", variant="primary", scale=1, elem_id="gen_btn")
            
            output_gallery = gr.Gallery(label="Output Gallery", columns=2, object_fit="contain", height=480)
            output_time = gr.Markdown("Processing Time: -- sec")
            
            gr.Markdown("---")
            gr.Markdown("### Drag & Drop image to restore settings")
            input_image_for_info = gr.Image(label="Restore Settings from PNG", type="pil", height=160,sources=["upload"])

    # リフレッシュボタンを押した時の動作
    def update_lists():
        return gr.update(choices=get_model_list()), gr.update(choices=get_lora_list())
    refresh_btn.click(fn=update_lists, inputs=[], outputs=[model_drop, lora_drop])

    gen_btn.click(
        fn=prepare_generation,
        inputs=[seed, seed_mode],
        outputs=[seed, start_time_state]
    ).then(
        fn=generate, 
        # 入力に lora_drop, lora_weight を追加
        inputs=[model_drop, lora_drop, lora_weight, prompt, neg_prompt, width, height, steps, cfg_scale, seed, seed_mode, batch_count, device_radio], 
        outputs=[output_gallery, save_path_state, final_seed_state] 
    ).then(
        fn=lambda start, paths, f_seed: (
            f"**Processing Time : {time.time() - start:.2f} sec**",
            f_seed
        ),
        inputs=[start_time_state, save_path_state, final_seed_state],
        outputs=[output_time, seed] 
    )

    input_image_for_info.change(
        fn=load_settings_from_image,
        inputs=[input_image_for_info],
        # 出力に lora_drop, lora_weight を追加
        outputs=[model_drop, lora_drop, lora_weight, prompt, neg_prompt, width, height, steps, cfg_scale, seed, seed_mode]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, css="#gen_btn { height: 80px !important; }")
