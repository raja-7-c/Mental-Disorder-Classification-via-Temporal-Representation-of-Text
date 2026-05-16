# Mental Disorder Classification via Temporal Representation of Text
<p align="center">
  <a href="https://arxiv.org/abs/2406.15470"><img src="https://img.shields.io/badge/arXiv-2406.15470-b31b1b.svg" alt="arXiv"></a>
  <a href="https://aclanthology.org/2024.findings-emnlp.639/"><img src="https://img.shields.io/badge/EMNLP-2024-blue.svg" alt="EMNLP 2024"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License"></a>
</p>

## Introduction 

Mental disorders pose a global challenge, aggravated by the shortage of qualified mental health professionals. Mental disorder prediction from social media posts by current LLMs is challenging due to the complexities of sequential text data and the limited context length of language models. Current language model-based approaches split a single data instance into multiple chunks to compensate for limited context size. The predictive model is then applied to each chunk individually, and the most voted output is selected as the final prediction. This results in the loss of inter-post dependencies and important time variant information, leading to poor performance. We propose a novel framework which first compresses the large sequence of chronologically ordered social media posts into a series of numbers. We then use this time variant representation for mental disorder classification. We demonstrate the generalization capabilities of our framework by outperforming the current SOTA in three different mental conditions: depression, self-harm, and anorexia, with an absolute improvement of 5% in the F1 score. We investigate the situation where current data instances fall within the context length of language models and present empirical results highlighting the importance of temporal properties of textual data. Furthermore, we utilize the proposed framework for a cross-domain study, exploring commonalities across disorders and the possibility of inter-domain data usage.
***

## The Problem with Existing Approaches
 
Current LLM-based approaches for mental disorder detection from social media suffer from three fundamental issues due to the **limited context length** of language models:
 
| Issue | Description |
|---|---|
| **Loss of temporal information** | Posts are split into chunks and processed independently, destroying chronological order and inter-post dependencies. |
| **Lack of global view** | Majority voting across chunks can misclassify users whose disorder-indicative posts are concentrated in only a few chunks. |
| **Semantic noise** | Concatenating posts with different semantic content into a single chunk dilutes the signal of individual high-importance posts. |
 
For example, a post indicating severe depression concatenated with a subsequent post about an unrelated topic introduces noise that weakens the predictive signal.

<img width="475" height="331" alt="comparing_approaches" src="https://github.com/user-attachments/assets/35853185-57df-48f7-be9f-f4ba63ff1666" />


## Proposed Framework
The proposed approach is a three-stage pipeline that transforms a user's chronological post history into a time series, then classifies it.
<img width="5064" height="1992" alt="Mental_Disorder_Temporal_Representation_architecture" src="https://github.com/user-attachments/assets/6f479ed6-3b84-4a28-bde4-048b768bc679" />
***



If you find this work, framework, or code useful in your research, please cite our paper:

```
@inproceedings{kumar-etal-2024-mental,
    title = "Mental Disorder Classification via Temporal Representation of Text",
    author = "Kumar, Raja and Maharaj, Kishan and Saxena, Ashita and Bhattacharyya, Pushpak",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    pages = "10901--10916",
    url = "[https://aclanthology.org/2024.findings-emnlp.639](https://aclanthology.org/2024.findings-emnlp.639)",
    doi = "10.48550/arXiv.2406.15470"
```


---
