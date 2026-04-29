import gradio as gr
import os
import subprocess
import sys
import urllib.request #

# フォルダの設定
BASE_MODEL_DIR = "./base_models"
STORAGE_DIR = "./storage"
MODELS_DIR = "./models"

os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 足りないスクリプトを自動で拾ってくる関数
def ensure_conversion_script():
    script_path = os.path.abspath(os.path.join(STORAGE_DIR, "convert_original_stable_diffusion_to_diffusers.py"))
    if not os.path.exists(script_path):
        print(f"変換スクリプトが見つかりません。公式からダウンロードします...")
        url = "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py"
        try:
            urllib.request.urlretrieve(url, script_path)
            print(f"ダウンロード完了: {script_path}")
        except Exception as e:
            raise RuntimeError(f"スクリプトのダウンロードに失敗しました: {e}")
    return script_path

def get_model_list():
    return [f for f in os.listdir(BASE_MODEL_DIR) if f.endswith('.safetensors')]

def run_command(cmd_list):
    process = subprocess.Popen(
        cmd_list, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )
    for line in process.stdout:
        yield line
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"コマンド実行エラー (終了コード: {process.returncode})")

def convert_to_diffusers_hybrid(checkpoint_path, dump_path):
    # スクリプトを確認し、絶対パスを取得
    old_script_path = ensure_conversion_script()

    py_code = f"""
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
from transformers import CLIPTextModelWithProjection, T5EncoderModel
import os
import subprocess
import sys

ckpt_path = os.path.abspath("{checkpoint_path}")
dump_path = os.path.abspath("{dump_path}")
old_script = os.path.abspath("{old_script_path}")

print("モデルのアーキテクチャを解析しています...")

model_type = "SD1.5" 
try:
    from safetensors import safe_open
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        if any("joint_blocks" in k for k in keys):
            model_type = "SD3.5"
        elif any("conditioner.embedders.1" in k for k in keys):
            model_type = "SDXL"
    print(f"判定結果: このモデルは【{{model_type}}】です。")
except Exception as e:
    print(f"判定失敗( {{e}} )。デフォルトの挙動で進めます。")

if model_type == "SD3.5":
    print("SD3.5で変換を開始します...")
    pipe = StableDiffusion3Pipeline.from_single_file(
        ckpt_path,
        text_encoder=CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="text_encoder", torch_dtype=torch.float16),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="text_encoder_2", torch_dtype=torch.float16),
        text_encoder_3=T5EncoderModel.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="text_encoder_3", torch_dtype=torch.float16),
        torch_dtype=torch.float16
    )
    pipe.save_pretrained(dump_path, safe_serialization=True)

else:
    print(f"{{model_type}} 、COSDDで変換を開始します...")
    cmd = [
        sys.executable, old_script,
        "--checkpoint_path", ckpt_path,
        "--dump_path", dump_path,
        "--from_safetensors",
        "--to_safetensors",
        "--device", "cpu"
    ]
    if model_type == "SDXL":
        cmd.append("--half")

    subprocess.run(cmd, check=True)

print("Step 1: 完了しました。")
"""
    tmp_script = os.path.join(STORAGE_DIR, "_tmp_convert.py")
    with open(tmp_script, "w", encoding="utf-8") as f:
        f.write(py_code)
    
    for output in run_command([sys.executable, tmp_script]):
        yield output

def convert_model(model_filename, precision):
    if not model_filename:
        yield "❌ エラー: モデルが選択されていません。"
        return

    log_text = ""
    def append_log(text):
        nonlocal log_text
        log_text += text
        return log_text

    model_path = os.path.join(BASE_MODEL_DIR, model_filename)
    model_name = os.path.splitext(model_filename)[0]
    dif_model_dir = os.path.join(STORAGE_DIR, f"{model_name}-diffusers")
    ov_model_dir = os.path.join(MODELS_DIR, f"{model_name}-{precision}-ov")

    try:
        # Step 1: 変換（SD3.5かどうかで手法を自動切り替え）
        if not os.path.exists(dif_model_dir):
            yield append_log(f"--- Step 1: 変換開始 ---\n")
            for output in convert_to_diffusers_hybrid(model_path, dif_model_dir):
                yield append_log(output)
        else:
            yield append_log(f"Step 1: 既存のモデルを再利用します。\n")

        # Step 2: OpenVINO形式への変換
        yield append_log(f"--- Step 2: OpenVINO形式へ変換開始 (精度: {precision}) ---\n")
        cmd_step2 = ["optimum-cli", "export", "openvino", "--model", dif_model_dir, "--task", "text-to-image", "--weight-format", precision]
        if precision == "int4": cmd_step2.extend(["--group-size", "64"])
        cmd_step2.append(ov_model_dir)

        for output in run_command(cmd_step2):
            yield append_log(output)

        yield append_log(f"\n全ての変換が完了しました！\n保存先: {ov_model_dir}\n")

    except Exception as e:
        yield append_log(f"\n処理が中断されました: {e}\n")

# Gradio UI
with gr.Blocks(title="OpenVINO Model Converter") as demo:
    gr.Markdown("# OpenVINO Model Converter GUI")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=get_model_list(), label="Target Model", scale=4)
                refresh_btn = gr.Button("更新", scale=1)
            precision_radio = gr.Radio(choices=["fp16", "int8", "int4"], value="int8", label="Conversion Precision")
            convert_btn = gr.Button("変換開始", variant="primary")
        with gr.Column(scale=1):
            console_log = gr.Textbox(label="Terminal Log", lines=25, max_lines=25, interactive=False, autoscroll=True)

    refresh_btn.click(fn=lambda: gr.update(choices=get_model_list()), inputs=[], outputs=[model_dropdown])
    convert_btn.click(fn=convert_model, inputs=[model_dropdown, precision_radio], outputs=[console_log])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
