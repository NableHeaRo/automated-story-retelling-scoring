# auto-SR: Automated Scoring of Story Retelling

Automated scoring of story-retelling (narrative discourse) performance using
sentence-embedding semantic similarity, as described in:

> Yoo, Y. R., Lim, Y., & Sung, J. E. (2026). Automated Scoring of Semantic Recall and Temporal Organization in Aphasic Story Retelling Using Language Model Embeddings
> *American Journal of Speech-Language Pathology*. [add DOI/volume once available]

Given (1) the original story text and (2) a person's retelling transcript,
this tool computes two automated outcome measures:

- **ORS (Overall Recall Score)** — how much of the story's content was
  recalled *somewhere* in the retelling, regardless of order.
- **OORS (Original-Order Recall Score)** — how well the story's original
  temporal sequence was preserved during recall.

and produces a semantic similarity matrix heatmap (and, for batches of
participants, a serial-position plot across story segments).

This is a research tool intended to support discourse assessment research
and is **not** a clinical diagnostic instrument.

## How it works

1. **Segmentation** — the story and the recall transcript are each split
   into 10 overlapping word-based segments (20% overlap), following
   Herrmann (2025).
2. **Embedding** — each segment is encoded into a 768-dimensional vector
   using the Language-agnostic BERT Sentence Embedding model, LaBSE
   (Feng et al., 2022), which supports 100+ languages.
3. **Similarity matrix** — pairwise Spearman correlations between all
   story-segment and recall-segment embeddings are computed and
   Fisher z-transformed, producing a 10 × 10 matrix.
4. **Outcome measures** — ORS is the average of each row's maximum value;
   OORS is the average of the matrix diagonal.

## Installation

```bash
git clone https://github.com/NableHeaRo/automated-story-retelling-scoring.git
cd automated-story-retelling-scoring
pip install -r requirements.txt
```

Requires Python 3.9+. The first run will download the LaBSE model
(~1.8 GB) from Hugging Face, so an internet connection is needed the
first time.

## Usage

> The files under `examples/` are placeholders showing the expected
> format — replace their contents with your own story stimulus and
> transcripts before running anything for real.

### Option A: Python API

```python
from auto_sr import score_retelling

story_text = open("examples/example_story.txt", encoding="utf-8").read()  # write your own stimulus here
recall_text = open("examples/example_recall.txt", encoding="utf-8").read()  # write your own transcript here

result = score_retelling(story_text, recall_text, plot=True, plot_save_path="matrix.png")

print("ORS:", result["ORS"])
print("OORS:", result["OORS"])
# result["z_matrix"] is the 10x10 Fisher-z similarity matrix (numpy array)
```

If you are scoring many recalls against the same story, reuse a single
embedder so the model is only loaded once:

```python
from auto_sr import LaBSEEmbedder, score_retelling

embedder = LaBSEEmbedder()
for pid, recall_text in recalls.items():
    result = score_retelling(story_text, recall_text, embedder=embedder)
    print(pid, result["ORS"], result["OORS"])
```

### Option B: Command line

Score a single participant:

```bash
python scripts/run_scoring.py single \
    --story examples/example_story.txt \
    --recall examples/example_recall.txt \
    --id P01 \
    --outdir results/
```

This saves, into `results/`:
- `P01_scores.csv` — ORS and OORS
- `P01_z_matrix.npy` / `.csv` — the full similarity matrix
- `P01_similarity_matrix.png` — the heatmap

Score a batch of participants from a CSV (columns: `id`, `recall_text`,
optional `group`):

```bash
python scripts/run_scoring.py batch \
    --story examples/example_story.txt \
    --recall-csv examples/example_recalls.csv \
    --outdir results/
```

This additionally saves `summary_scores.csv` (all participants) and
`serial_position_plot.png` (diagonal similarity by story segment, with
group means if a `group` column was provided).

## Input format notes

- **Story and recall text** should be plain text, one clause/sentence per
  line or space-separated — segmentation is based on whitespace-separated
  word counts, so any consistent tokenization works.
- **Transcription**: this tool does not perform speech-to-text. Recall
  transcripts should be prepared beforehand (e.g., via manual
  transcription or any speech-to-text service) and manually checked for
  transcription accuracy before scoring, as in the published study.
- LaBSE is multilingual, so this pipeline is not limited to Korean, but
  ORS/OORS have only been validated against manual scoring in the Korean
  aphasia sample reported in the paper above. Validation in other
  languages/populations is an open question for future work.

## Citation

If you use this code, please cite all of the following:

```bibtex
@article{yoo2026autosr,
  title   = {Automated Scoring of Semantic Recall and Temporal Organization in Aphasic Story Retelling Using Language Model Embeddings},
  author  = {Yoo, Yae Rin and Lim, Yoonseob and Sung, Jee Eun},
  journal = {American Journal of Speech-Language Pathology},
  year    = {2026}
}

@article{herrmann2025autosr,
  title   = {Language-agnostic, automated assessment of listeners' speech recall using large language models},
  author  = {Herrmann, Bj{\"o}rn},
  journal = {arXiv preprint arXiv:2503.01045},
  year    = {2025}
}

@article{feng2022labse,
  title   = {Language-agnostic BERT sentence embedding},
  author  = {Feng, Fangxiaoyu and Yang, Yinfei and Cer, Daniel and Arivazhagan, Naveen and Wang, Wei},
  journal = {arXiv preprint arXiv:2207.04672},
  year    = {2022}
}
```

## License

[TBD]

## Authors

Yae Rin Yoo^a,b^, Yoonseob Lim^b^, Jee Eun Sung^a,c,d*^

^a^ Department of Communication Disorders, Ewha Womans University, Seoul, Republic of Korea
^b^ Intelligence and Interaction Research Center, Korea Institute of Science and Technology, Seoul, Republic of Korea
^c^ Department of Brain and Cognitive Sciences, Ewha Womans University, Seoul, Republic of Korea
^d^ Ewha Brain Institute, Ewha Womans University, Seoul, Republic of Korea

## Contact

Yae Rin Yoo — leader9390@ewhain.net
