# ComfyUI InstantID (原生支持)

[InstantID](https://github.com/InstantID/InstantID) 的原生 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 支持。

此扩展不同于许多已可用的扩展，因为它不使用 *diffusers*，而是原生实现了 InstantID，并且与 ComfyUI 完全集成。

# 赞助

<div align="center">

**[:heart: Github 赞助](https://github.com/sponsors/cubiq) | [:coin: Paypal](https://paypal.me/matt3o)**

</div>

如果您喜欢我的工作并希望看到更新和新功能，请考虑赞助我的项目。

- [ComfyUI IPAdapter Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [ComfyUI InstantID (原生)](https://github.com/cubiq/ComfyUI_InstantID)
- [ComfyUI Essentials](https://github.com/cubiq/ComfyUI_essentials)
- [ComfyUI FaceAnalysis](https://github.com/cubiq/ComfyUI_FaceAnalysis)

更不用说文档和视频教程。可以查看我在 YouTube 上的 **ComfyUI 高级理解** 视频，例如 [第 1 部分](https://www.youtube.com/watch?v=_C7kR2TFIX0) 和 [第 2 部分](https://www.youtube.com/watch?v=ijqXnW_9gzc)。

保持代码开源和免费的唯一方法是通过赞助其开发。赞助越多，我就能投入更多时间在我的开源项目上。

请考虑 [Github 赞助](https://github.com/sponsors/cubiq) 或 [PayPal 捐赠](https://paypal.me/matt3o)（Matteo "matt3o" Spinelli）。对于赞助 $50+ 的人，请告诉我是否希望在此 README 文件中被提及，您可以在 [Discord](https://latent.vision/discord) 或通过 _matt3o :snail: gmail.com_ 联系我。

## 重要更新

- **2024/02/27:** 在负嵌入中添加了[噪声注入](#noise-injection)。

- **2024/02/26:** 修复了一个小但讨厌的错误。结果将有所不同，您可能需要降低 CFG。

- **2024/02/20:** 我重构了节点，希望它们更易于使用。**这是一次重大更新**，以前的工作流将不再可用。

## 基本工作流

在 `examples` 目录中，您会找到一些基本工作流。

![workflow](examples/instantid_basic_workflow.jpg)

## 视频教程

<a href="https://youtu.be/wMLiGhogOPE" target="_blank">
 <img src="https://img.youtube.com/vi/wMLiGhogOPE/hqdefault.jpg" alt="观看视频" />
</a>

** :movie_camera: [InstantID 功能介绍](https://youtu.be/wMLiGhogOPE)**

## 安装

**将 ComfyUI 升级到最新版本！**

下载或 `git clone` 此仓库到 `ComfyUI/custom_nodes/` 目录或使用 Manager。

InstantID 需要 `insightface`，您需要将其添加到您的库中，连同 `onnxruntime` 和 `onnxruntime-gpu`。

InsightFace 模型是 **antelopev2**（不是经典的 buffalo_l）。下载模型（例如从 [这里](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) 或 [这里](https://huggingface.co/MonsterMMORPG/tools/tree/main)），解压并将其放置在 `ComfyUI/models/insightface/models/antelopev2` 目录中。

**主模型**可以从 [HuggingFace](https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true) 下载，应将其放置在 `ComfyUI/models/instantid` 目录中。（请注意，该模型称为 *ip_adapter*，因为它基于 [IPAdapter](https://github.com/tencent-ailab/IP-Adapter)）。

您还需要一个 [controlnet](https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true)，将其放置在 ComfyUI controlnet 目录中。

**请记住，目前这仅适用于 SDXL。**

## 水印！

训练数据中充满了水印，为避免水印出现在您的生成中，请使用与 1024×1024（或标准尺寸）略有不同的分辨率，例如 **1016×1016** 效果很好。

## 降低 CFG！

重要的是将 CFG 降低到至少 4/5，或者您可以使用 `RescaleCFG` 节点。

## 面部关键点

人物的姿势是基于从参考图像生成的关键点。您可以通过向 `image_kps` 输入发送图像来使用不同的姿势。

<img src="examples/daydreaming.jpg" width="386" height="386" alt="白日梦" />

## 噪声注入

默认的 InstantID 实现似乎真的“烧坏”了图像，我发现通过向负嵌入中注入噪声，我们可以缓解这一效果，并增加与参考的相似性。默认的 Apply InstantID 节点自动注入 35% 的噪声，如果您想微调效果，可以使用 Advanced InstantID 节点。

这仍然是实验性的，可能会在未来发生变化。

## 额外的 Controlnets

您可以向生成中添加更多 controlnets。提供了一个用于深度 controlnet 的示例工作流。

## 使用 IPAdapter 进行样式化

可以使用 IPAdapter 对构图进行样式化。提供了一个示例。

<img src="examples/instant_id_ipadapter.jpg" width="512" alt="IPAdapter" />

## 多-ID 支持

支持多 ID，但工作流有点复杂，生成速度较慢。我会检查是否可以找到更好的方法。示例工作流在 examples 目录中提供。

<img src="examples/instantid_multi_id.jpg" width="768" alt="IPAdapter" />

## 高级节点

目前有一个高级的 InstantID 节点，当前与标准节点的唯一区别是您可以分别设置 instantID 模型和 controlnet 的权重。它现在还包括一个噪声注入选项。对于微调可能很有帮助。

instantID 模型对构图的影响约为 25%，其余的是 controlnet。

噪声有助于减少“燃烧”效果。

## 其他注意事项

它与 SDXL Turbo/Lighting 非常兼容。使用社区的检查点效果最好。

## 当前赞助商

正是由于慷慨的赞助商，**整个社区**才能享受开源和免费软件。请与我一起感谢以下公司和个人！

### :trophy: 金牌赞助商

[![Kaiber.ai](https://f.latent.vision/imgs/kaiber.png)](https://kaiber.ai/)&nbsp; &nbsp;[![InstaSD](https://f.latent.vision/imgs/instasd.png)](https://www.instasd.com/)

### :tada: 银牌赞助商

[![OperArt.ai](https://f.latent.vision/imgs/openart.png?r=1)](https://openart.ai/workflows)&nbsp; &nbsp;[![Finetuners](https://f.latent.vision/imgs/finetuners.png)](https://www.finetuners.ai/)&nbsp; &nbsp;[![Comfy.ICU](https://f.latent.vision/imgs/comfyicu.png?r=1)](https://comfy.icu/)

### 其他支持我项目的公司

- [RunComfy](https://www.runcomfy.com/) (ComfyUI 云) 

### 尊敬的个人

- [Øystein Ø. Olsen](https://github.com/FireNeslo)
- [Jack Gane](https://github.com/ganeJackS)
- [Nathan Shipley](https://www.nathanshipley.com/)
- [Dkdnzia](https://github.com/Dkdnzia)

[以及所有我的公开和私密赞助商！](https://github.com/sponsors/cubiq)
