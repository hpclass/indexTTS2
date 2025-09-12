
<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>


<h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>

## 👉🏻 IndexTTS2 👈🏻

[[HuggingFace Demo]](https://huggingface.co/spaces/IndexTeam/IndexTTS)   [[ModelScope Demo]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo) \
[[Paper]](https://arxiv.org/abs/2502.05512)  [[Demos]](https://index-tts.github.io/index-tts2.github.io/)  

Large-scale text-to-speech (TTS)  models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of the synthesized speech. This becomes a significant limitation in applications such as video dubbing, where strict audio-visual synchronization is required. This paper introduces IndexTTS2, which proposes a novel, general, and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens, thereby enabling precise control over speech duration; the other does not require manual token count input, letting the model freely generate speech in an autoregressive manner while faithfully reproducing prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model is capable of perfectly reproducing the emotional characteristics inherent in the input prompt. Additionally, users may provide a separate emotion prompt (which can originate from a different speaker than the timbre prompt), thereby enabling the model to accurately reconstruct the target timbre while conveying the specified emotional tone. In order to enhance the clarity of speech during strong emotional expressions, we incorporate GPT latent representations to improve the stability of the generated speech. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This facilitates the effective guidance of speech generation with the desired emotional tendencies through natural language input. Finally, experimental results on multiple datasets demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in terms of word error rate, speaker similarity, and emotional fidelity. To promote further research and facilitate practical adoption, we will release both the model weights and inference code, enabling the community to reproduce and build upon our work.
<span style="font-size:16px;">  
Experience **IndexTTS**: Please contact <u>zhousiyi02@bilibili.com</u> <u>zhouxun@bilibili.com</u> for more detailed information. </span>
### Contact
QQ群（二群）：1048202584 \
Discord：https://discord.gg/uT32E7KDmy  \
简历：indexspeech@bilibili.com  \
欢迎大家来交流讨论！
## 📣 Updates

- `2025/06/10` 🔥🔥🔥  We release the **IndexTTS2**
    - The first autoregressive TTS model with precise synthesis duration control: supporting both controllable and uncontrollable modes 
    - The model achieves highly expressive emotional speech synthesis, with emotion-controllable capabilities enabled through multiple input modalities.
- `2025/05/14` 🔥🔥 We release the **IndexTTS-1.5**, Significantly improve the model's stability and its performance in the English language.
- `2025/03/25` 🔥 We release IndexTTS-1.0 model parameters and inference code.
- `2025/02/12` 🔥 We submitted our paper on arXiv, and released our demos and test sets.

## 🖥️ Method

The overview of IndexTTS is shown as follows.

<picture>
  <img src="assets/IndexTTS.png"  width="800"/>
</picture>


The key contributions of **indextts2** are summarized as follows:
 - We propose a novel and heuristic duration adaptation scheme for autoregressive large-scale TTS models. IndexTTS2 is the first autoregressive zero-shot TTS model that combines precise duration control with free generation of natural durations, striking a balance between flexibility, controllability and autoregressive nature.
 - The emotional and speaker-related features are decoupled from the prompts, and a feature fusion strategy is designed to maintain semantic fluency and pronunciation clarity during emotionally rich expressions. Furthermore, a tool was developed for emotion control, utilising natural language descriptions for the benefit of users.
 - We publicly release the code and pre-trained weights to facilitate future research and practical applications.



## Model Download
| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁 IndexTTS2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |


## Usage Instructions
### Environment Setup
1. Download this repository:
```bash
git clone https://github.com/index-tts/index-tts.git
```
2. Install dependencies:
```bash
conda create -n indextts2 python=3.10
conda activate indextts2
pip install -r requirements.txt
```

3. Download models:

Download by `huggingface-cli`:

```bash
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
```

Recommended for China users. 如果下载速度慢，可以使用镜像：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Or by `wget`:

```bash
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P checkpoints
```

4. Run test script:

Do a quick test run

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", verbose=True)
```

额外指定一个情感参考音频 Specify an additional emotional reference audio

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", emo_audio_prompt="test_data/low.wav", verbose=True)
```

当指定情感参考音频时，还可以额外指定参数emo_alpha，emo_alpha代表参考情感音频的程度，默认为1.0

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", emo_audio_prompt="test_data/low.wav", emo_alpha=0.5, verbose=True)
```


也可以不指定情感参考音频，而给定各基础情感(喜|怒|哀|惧|厌恶|低落|惊喜|平静)的强度，包括8个float的list

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", emo_vector=[0, 1.0, 0, 0, 0, 0, 0, 0], verbose=True)
```

可以使用文本情感描述指导情感的合成，使用参数use_emo_text

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", use_emo_text=True, verbose=True)
```

当不指定emo_text，根据输入的合成文案内容推理，指定时根据指定的文案推

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", use_emo_text=True, emo_text='有一丢丢伤心', verbose=True)
```



Specify the duration of the synthesized speech

```bash
from indextts.infer_indextts2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, use_cuda_kernel=False)
text="这是一个有很好情感表现力的自回归TTS大模型，它还可以控制合成语音的时长，希望能受到大家的喜欢。"
tts.infer(spk_audio_prompt='test_data/input.wav', text=text, output_path="gen.wav", use_speed=True, target_dur=7.5, verbose=True)
```


5. Use as command line tool:

```bash
# Make sure pytorch has been installed before running this command
pip install -e .
indextts "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！" \
  --voice reference_voice.wav \
  --model_dir checkpoints \
  --config checkpoints/config.yaml \
  --output output.wav
```

Use `--help` to see more options.
```bash
indextts --help
```

#### Web Demo
```bash
pip install -e ".[webui]"
python webui.py

# use another model version:
python webui.py --model_dir IndexTTS-1.5
```
Open your browser and visit `http://127.0.0.1:7860` to see the demo.

#### Note for Windows Users

On Windows, you may encounter [an error](https://github.com/index-tts/index-tts/issues/61) when installing `pynini`:
`ERROR: Failed building wheel for pynini`

In this case, please install `pynini` via `conda`:

```bash
# after conda activate index-tts
conda install -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3
pip install -e ".[webui]"
```

#### Sample Code
```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice="reference_voice.wav"
text="大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, output_path)
```

## Acknowledge
1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)

## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
