# Query‑Based Object‑Centric Video Summarizer

**Goal**  
Turn a raw video + a natural‑language query into  

1. **A short, human‑readable summary sentence** (“The dog catches a frisbee and runs away.”)  
2. **Spatio‑temporal masks / boxes** for the queried object through the whole clip  

---

## 1 Big‑Picture Overview

| | |
|---|---|
| **Problem** | Existing pipelines either summarise *whole* videos without object focus **or** track objects without textual explanation. We combine **grounded detection → tracking → language** so users can *see* what the summary refers to. |
| **Input** | `video.mp4`, query string (e.g. `"dog"`, `"red car"`). |
| **Output** | `(summary.txt, masks.npy)` or an annotated GIF. |
| **Upstream vs. Downstream** | *Upstream* modules (feature extractors) feed a *downstream* summariser task. |

---

## 2 Key Components

### 2.1 Models

| Stage | Model | Role |
|-------|-------|------|
| Grounded detection | **Grounding‑DINO** | Query‑conditioned bounding boxes per frame. |
| Masking | **Segment Anything (SAM)** | Box → high‑quality mask. |
| Tracking | **Track‑Anything** (SAM + XMem) | Propagate the mask with re‑ID & occlusion handling. |
| Language | **BLIP‑2** | Produce a one‑sentence caption for the masked clip. |

### 2.2 Datasets

| Dataset | Used for |
|---------|---------|
| **DAVIS 2017** | IoU / MOTA evaluation of masks. |
| **VidSTG** | Query‑caption supervision (BLEU/CIDEr). |
| *(optional)* SumMe, TVSum | Extra summarisation pre‑training. |

### 2.3 Metrics

| Sub‑task | Metric |
|----------|--------|
| Grounding | IoU / AP |
| Tracking | MOTA, IDF1 |
| Summarisation | BLEU‑4, METEOR, CIDEr |

---

## 3 End‑to‑End Pipeline


1. **Grounding**: detect query object each frame.  
2. **Masking**: refine first box to a mask.  
3. **Tracking**: propagate mask through clip.  
4. **Summarising**: feed masked frames + query to BLIP‑2.  
5. **Evaluate**: compute IoU/MOTA + BLEU/CIDEr on DAVIS & VidSTG.

---

## 4 Setup and Weight Initialisation

* We cloned the necessary models from the repositories below. To run our model you first have to set up the models' weights using the commands below. (Weight files are too big to upload it into github unfortunately). Also you can always visit the model repos if you have to reclone these models.

## Clone Grounding DINO
[https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

```
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## Clone SAM
[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

````
mkdir sam_initial_checkpoint
wget -O sam_initial_checkpoint/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
````

## Clone TrackAnything
[https://github.com/gaomingqi/Track-Anything](https://github.com/gaomingqi/Track-Anything)

## Clone BLIP2
[https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## Clone SAMTrack
[https://github.com/z-x-yang/Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)

