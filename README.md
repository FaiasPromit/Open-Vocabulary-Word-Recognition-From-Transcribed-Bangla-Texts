# Open Vocabulary Word Recognition From Transcribed Bangla Texts

Code for the paper **"Open Vocabulary Word Recognition From Transcribed Bangla Texts"** (ICCIT 2023).

Paper: [IEEE](https://doi.org/10.1109/ICCIT60459.2023.10441393) · [arXiv](https://arxiv.org/abs/2610.01134)  
Thesis and related papers: [Optical Character Recognition From Handwritten Bangla Texts](https://github.com/FaiasPromit/Optical-Character-Recognition-From-Handwritten-Bangla-Texts)

The code recognizes handwritten Bangla words by detecting each character with two object detection models, SSD with MobileNetV2 and Faster R-CNN with InceptionResNetV2, and combining them in an ensemble. Because it works character by character, it is not limited to a fixed vocabulary.

## How to run

The code runs in Google Colab.

1. Download the files from this repository.
2. Upload them to the top level of your Google Drive (My Drive).
3. Unzip `Zip Folder of Models`.
4. Move the `models(1)` folder into the same folder as `CustomTF2`.
5. Open the notebook you need in Google Colab and run it.

## Notes before you start

- **Training:** each training notebook includes step-by-step instructions. Follow them. The main files you need are already provided.
- **Testing:** the test notebooks are not meant to be run top to bottom. Read the note above each cell to see whether to run it or skip it.

## Dataset

The full dataset, PromitoLipi, is available on [Mendeley Data](https://data.mendeley.com/datasets/fnw59h7y89/2). The word images and their annotations are in the `PromitoLipi2.1` folder.

## Citation

If you use this code, please cite:

F. Satter and S. M. Masudul Ahsan, "Open Vocabulary Word Recognition From Transcribed Bangla Texts," *2023 26th International Conference on Computer and Information Technology (ICCIT)*, Cox's Bazar, Bangladesh, 2023, pp. 1-6, doi: 10.1109/ICCIT60459.2023.10441393.

```bibtex
@inproceedings{satter2023open,
  author    = {Satter, Faias and Ahsan, Sk. Md. Masudul},
  title     = {Open Vocabulary Word Recognition From Transcribed Bangla Texts},
  booktitle = {2023 26th International Conference on Computer and Information Technology (ICCIT)},
  address   = {Cox's Bazar, Bangladesh},
  year      = {2023},
  pages     = {1--6},
  doi       = {10.1109/ICCIT60459.2023.10441393}
}
```
