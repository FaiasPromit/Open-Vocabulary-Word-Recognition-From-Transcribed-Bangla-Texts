# Open Vocabulary Word Recognition From Transcribed Bangla Texts

Code for the ICCIT 2023 paper by Faias Satter and Sk. Md. Masudul Ahsan.

[arXiv](https://arxiv.org/abs/2610.01134) · [IEEE](https://doi.org/10.1109/ICCIT60459.2023.10441393) · [Thesis](https://github.com/FaiasPromit/Optical-Character-Recognition-From-Handwritten-Bangla-Texts)

## Setup

Download [PromitoLipi2.1](https://data.mendeley.com/datasets/fnw59h7y89/2). The TensorFlow models checkout, pipeline configs, trained weights and train/test split must be supplied separately.

Open the notebook in Jupyter or Colab and edit its configuration cell. Set `USE_GOOGLE_DRIVE = True` to mount Drive, or leave it `False` for local paths. The default folders beneath `BASE_DIR` are:

| Path | Contents |
|---|---|
| `models/research` | TensorFlow Object Detection API checkout |
| `customTF2/data/images` | Training/test images referenced by the CSVs |
| `customTF2/data/train_labels`, `test_labels` | Training/test XML annotations |
| `customTF2/data/Test` | Test BMP images and XMLs with matching filenames |
| `customTF2/data/label_map.pbtxt` | Numeric class names and IDs 1–92 |
| `customTF2/data/inference_graph/saved_model` | SSD model |
| `customTF2/data/inference_graph (1)/saved_model` | Faster R-CNN model |

`requirements-notebook.txt` lists supporting packages for Python 3.10 / TensorFlow 2.13. The two `requirements-*-historical.txt` files record versions found in the training and inference logs; they are partial dependency lists. Use separate environments for the two stacks.

For API setup, provide a compatible TensorFlow models checkout and `protoc`, then enable `INSTALL_OBJECT_DETECTION`. Restart the kernel afterward and set the flag back to `False`.

## Training

Use `Train_SSD.ipynb` or `Train_Faster_RCNN.ipynb`. Set the pipeline config, data and checkpoint paths, including paths inside the config itself. Place `generate_tfrecord.py` at the configured helper path.

Run cells in order. Set `GENERATE_RECORDS = False` to reuse existing records. Each model has a separate `MODEL_DIR`; an existing checkpoint resumes training. `RUN_TRAINING` and `RUN_EXPORT` control those stages. Checkpoint evaluation can wait for new checkpoints, so `RUN_EVALUATION` is off by default and can be run separately.

## Testing

Use `SSD_Test_Result.ipynb`, `Faster_RCNN_Test_Result.ipynb` or `Ensemble_Model_Test_Result.ipynb`.

Set `EXPERIMENT = 'full'` for the complete method. The single-model notebooks also support `no_pno` and `no_nms_pno`. Enable `RUN_WORD_EVALUATION` for Levenshtein scoring, then run cells in order.

Results are saved as `recognized.txt` and `results.json` in `OUTPUT_DIR`. Choose an empty output directory for each run. Character counters use the notebook's first-match IoU > 0.4 rule; a wrong-class match counts as FP without an additional FN.

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
