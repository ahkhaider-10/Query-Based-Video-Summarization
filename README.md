# Query-Based-Video-Summarization
Query-Based, Object-Focused Video Summarisation in Single and Multi-Object Scenes by Using Grounding DINO, Track Anything and BLIP2 Models

## 1 · Big‑picture overview
Goal	Description
Task	Query‑based, object‑focused video summarisation – given a natural‑language query about an object (“What is the dog doing?”) and a raw video, output 
(a) a short textual summary of that object’s behaviour and 
(b) The tracked object masks/bounding boxes through time.
Why	does existing work either 
(i) summarises whole videos without object focus, 
(ii) tracks objects without textual explanation, or 
(iii) produces “black‑box” summaries with no interpretable tracking. Your system combines grounded detection → tracking → language so users can see what the model is talking about. 

## 2  ·  Key components
### 2.1  Models
Stage	Model: What it contributes
Grounded detection	Grounding‑DINO	takes the text query + frame → returns query‑conditioned bounding boxes. 
Masking	SAM (Segment Anything)	converts each bounding box to a high‑quality object mask.
Tracking	Track‑Anything (SAM + XMem)	Propagates the mask through the video, handling occlusion & re‑identification.
Language	BLIP‑2 (vision‑LLM)	accepts a clip or sequence of masked frames → produces a concise behaviour caption.

### 2.2  Datasets
Purpose	Dataset	Notes
Object masks / IoU	DAVIS 2017 video segmentation: High‑quality masks for single objects.
Textual grounding & summary	VidSTG	Each clip is paired with a query sentence and a ground‑truth description.
(Optional extras)	SumMe / TVSum (generic summarisation)	For ablation or extra training.

### 2.3  Metrics
Sub‑task	Metric
Grounding	IoU / AP on the first frame
Tracking	MOTA, IDF1
Summarisation	BLEU‑4, METEOR, CIDEr

## 3  ·  End‑to‑end pipeline

┌──────────────┐
│  Raw video   │
└──────┬───────┘

       ▼  (per frame)
       
┌────────────────────┐
│ Grounding‑DINO     │
│  query + frame →   │
│  bboxes            │
└──────┬─────────────┘

       ▼
       
┌────────────────────┐
│ SAM                │
│  bbox → mask       │
└──────┬─────────────┘

       ▼   (first mask)
       
┌────────────────────┐
│ Track‑Anything     │
│  propagate mask    │
│  through video     │
└──────┬─────────────┘

       ▼   (masked clip)
       
┌────────────────────┐
│ BLIP‑2             │
│  masked frames →   │
│  summary text      │
└────────────────────┘

Grounding – run per frame; keep the mask with the highest text‑similarity logit.

Tracking – warm‑start Track‑Anything with that mask; output full video mask stack.

Cropping/focusing – feed the masked region (optionally the original + mask) to BLIP‑2.

Summary generation – BLIP‑2 returns a caption; optionally post‑process with an LLM prompt:
“Write a one‑sentence summary starting with ‘The <object> …’ ”

Evaluation loop – compute IoU, MOTA, BLEU/CIDEr vs. VidSTG references.

## 4  ·  Starter code skeleton
Below is a minimal repo structure (Python).

video_summarizer/
├── datasets/
│   ├── __init__.py
│   ├── davis.py          # loads DAVIS videos + masks
│   └── vidstg.py         # loads VidSTG clips + queries
├── models/
│   ├── grounding_dino_wrapper.py
│   ├── sam_wrapper.py
│   ├── track_anything_wrapper.py
│   └── blip2_wrapper.py
├── pipeline/
│   ├── tracker.py        # orchestrates detection → mask → tracking
│   └── summarizer.py     # calls BLIP‑2, post‑processes text
├── utils/
│   ├── metrics.py        # IoU, MOTA, BLEU, etc.
│   └── vis.py            # overlay masks / save GIFs
├── main.py               # CLI entry‑point
└── config.yaml           # paths, hyper‑params
