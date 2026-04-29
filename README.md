# OpenVINO_GenAI_Image_generation
OpenVINO_GenAIを利用した画像生成ソフトです。

## ✨ 主な機能
Stable DiffusionやComfyUIで使われるモデルファイル（safetensors）をOpenVINO(IR形式)用に変換するツール。<br>
変換したモデルファイルを使って、画像生成。<br>
VRAMから溢れた分は、自動でメインメモリに退避、クラッシュを防ぎます。

## 💻 動作環境
* **OS**: Linux (Ubuntu24.04.4, Ubuntu26.04)で確認しています。
* **Python**: 3.12
* **ハードウェア**: 
  * 推奨: Intel GPU

## 🛠 インストール方法
1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/toaru-ubuntu/OpenVINO_GenAI_Image_generation.git
   cd OpenVINO_GenAI_Image_generation
   
2. **uv仮想環境の作成と有効化**
    ```bash
    sudo apt install curl

    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    
    uv venv .venv --python 3.12
    source .venv/bin/activate
    
3. **必要なパッケージのインストール**
    ```bash
    uv pip install openvino-genai optimum[openvino] datasets diffusers omegaconf accelerate transformers gradio sentencepiece protobuf
    
4. **safetensorsをIR形式に変換** <br>
    base_modelsフォルダの中に変換したいsafetensorsを配置
    ```python
    python safetensors_converter_gui.py
    
5. **画像生成**<br>
仮想環境を有効化した状態で、メインスクリプトを実行します。
    ```python
    python OV_image_generation.py

# 注意事項
* **SD3.5のモデルについて**<br>
SD3.5系のモデルも変換可能ですが、<br>
・[Hugging Face](https://huggingface.co/)のアクセストークン<br>
・[SD3.5 Largeの公式ページ](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)でのライセンスへの同意<br>
以上２点が必要になります。

* **ライセンス**<br>
OpenVINO_GenAI_Image_generation自体はMITライセンスですが、<br>
OpenVINO GenAIについては[こちら](https://github.com/openvinotoolkit/openvino.genai)を確認してください。
