# `models/` —— 仓内模型素材

**这里的东西跟着仓库走，不跟数据根走。** 它不是作品数据，是可以重新拿到的权重与词表：
丢了只是「要重新下一遍」，不会丢作品。

所以它**故意不在 `data/` 里**——`data/` 是会被整个搬走/备份的数据根，把 1.8 GB 可重下的权重
塞进备份对象里，只会让每次备份多拷一份没人看的东西。

---

## 结构：按用途分目录

```
models/
├── ocr/
│   └── PaddleOCR-VL-1.6/    ← OCR 权重（约 1.8 GB / 25 个文件）
└── tokenizers/
    ├── qwen3_4b/            ← 出图工作流的文本编码器词表
    ├── qwen3vl_4b/          ← 同上（VL 版）
    └── qwen35_9b/           ← Qwen-Image 2.1 线的词表
```

| 目录 | 谁用 | 从哪来 | 怎么恢复 |
|---|---|---|---|
| `ocr/PaddleOCR-VL-1.6/` | 知识库 OCR 线（`scripts/corpus/ocr_vlm_local.py`、`ocr_vlm_fetch.py`） | `ocr_vlm_fetch.py --fetch` 从 HuggingFace 拉 | 重跑 `--fetch`；或用 `OCR_VLM_MODEL_PATH` 指到别处的同名权重 |
| `tokenizers/*/` | 三档出图工作流（`engine/tokenizer_hub.py` 的仓内落点） | 随仓库/同源词表文件 | 从上游模型仓补回同名文件 |

`models/` 全部内容都在 `.gitignore` 里（`models/ocr/`、`models/tokenizers/`）——**权重不进仓**，
克隆下来是空目录，按上表恢复。

---

## 为什么不分类就乱了

2026-11-07 之前 `models/` 根下是「两个 tokenizer 目录 + 一个 OCR 目录」混排，还有一版旧 OCR 权重
（`PaddleOCR-VL`，1.8 GB）与新版（`PaddleOCR-VL-1.6`）并排站着，只差一个版本号后缀。
看目录名分不出：

- 哪个是出图要的、哪个是 OCR 要的；
- 哪个是现行版、哪个是留着对照的旧版。

现在按**用途**一层：OCR 的进 `ocr/`，词表进 `tokenizers/`。新增素材时按这个分，
不要往根上摊。

---

## 和 `<dataRoot>/models` 不是一处（别搞混）

| | `models/`（本目录） | `<dataRoot>/models/`（通常就是 `data/models/`） |
|---|---|---|
| 内容 | 仓内素材：OCR 权重、词表 | model_manager 下载的几十 GB 出图权重（DiT / VAE / LoRA） |
| 谁的 | 仓库自带，锚定 `PROJECT_ROOT` | 运行时数据，跟数据根迁移 |
| 进仓 | 否（体积大） | 否（运行时数据） |
| 备份 | 不必（能重下） | 视作品需要（它属于「跑图环境」而非作品） |

两者同名但不是同一个目录——写代码时别拼错根。`models/` 的路径锚在
`Path(__file__).resolve().parents[3].parent`（= 仓库根），不读 `get_data_root()`。

---

## 该不该把某个东西放到 `data/` 下

判据只有一条：**丢了以后怎么办**。

- 算得回来 ⇒ `indexes/`
- 下载得到 ⇒ `models/`（就是这里）
- 仓库里有 ⇒ `corpus/`
- **丢了就没了 ⇒ `data/`**（作品数据、生成物、台账）

按这条过一遍本目录：OCR 权重和词表都能重新下载 ⇒ 留在 `models/`，**不搬去 `data/`**。
