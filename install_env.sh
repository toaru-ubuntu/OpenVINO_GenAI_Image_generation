#!/bin/bash

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install openvino-genai optimum[openvino] datasets diffusers omegaconf accelerate transformers gradio sentencepiece protobuf

echo "base_modelにsafetensorsを入れて、safetensors_converter_gui.pyでint4などに量子化変換してください。"
echo "Add safetensors to base_model and use safetensors_converter_gui.py to quantize them to int4 or another format."
echo ""
echo "そのあと、OV_image_generation.pyで画像生成できます。"
echo "you can generate the image using OV_image_generation.py."
