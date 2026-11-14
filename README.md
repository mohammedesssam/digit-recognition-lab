# Digit Recognition Lab

## Repository packaging

The actual Python training, API and test source is browsable in this repository.
Original data, trained models, plots and execution evidence are supplied in a supplementary checksummed base64 archive because ordinary binary uploads failed. No external download or account is needed to restore it.

After cloning or downloading this repository, run:

```bash
python restore_artifacts.py
cd restored/digit-recognition-lab
```

Then follow the original setup and test commands below from that restored directory. The archive SHA-256 is `40d632fe525a1eb8e73b166d9716a56dc91ac5c3e7aaeea946ca652afecd928a`. Restoration only writes into a new `restored` directory and does not execute models or install anything. The source files in the restored package match the inspected source; its original README predates publication.

Repository publication is not cloud deployment. The app was exercised locally; no cloud-hosted prediction endpoint is claimed. GitHub Actions and Docker were not executed. This is newly AI-assisted educational work, not proof of prior independent applicant experience.


AI-assisted educational image-classification project. A real trained **PyTorch MLP** recognizes UCI 8x8 digit block-count images; a scikit-learn SVM baseline is included and honestly performs better.

## Run and reproduce

Python 3.10 was tested. Use a fresh virtual environment.
```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-dev.txt
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000
python -m pytest -q
python -m app.train
```

`requirements-cpu.txt` adds the official CPU PyTorch package index. The exact trained PyTorch version is pinned in requirements. Package downloads may be large. API serves the included checkpoint on CPU. `torch.load(weights_only=True)` is used; do not load unknown model files regardless.

`/` shows an original held-out digit, its dataset label and the model prediction. `/example/{index}` accepts 0-1796. `/predict` takes 64 integer block counts in [0,16] and returns a digit/top-three softmax scores. It rejects blank or malformed images. Scores are **not calibrated confidence**. `/docs`, `/metrics`, `/health` are included.

## Data and fair evaluation

Original UCI Optical Recognition of Handwritten Digits, not MNIST. Each image comes from a normalized 32x32 bitmap summarized as 8x8 counts of active pixels in 4x4 blocks. Pixel values are divided by a fixed physical maximum, 16; no fitted test-set statistics.

The original dataset separates writers: 30 people contribute to the original training set and 13 different people to the original test set. Preserve that original separation. The 3,823 original training examples are split stratified with seed 42 into 3,058 fitting and 765 validation samples. The 1,797 original test examples remain held out. There are no writer IDs within the training file, so internal validation is sample-stratified, **not claimed writer-separated**. Test writers remain separate per the dataset description.

Model: 64 inputs -> 64 ReLU hidden units -> 10 logits. Adam, learning rate .003, weight decay .001, full-batch 250 epochs. Choose the earliest checkpoint with maximum validation macro-F1, epoch 196. Cross-entropy loss, target labels 0-9. Fixed SVM baseline: C=10, gamma=scale. Test is evaluated only after fitting/checkpoint selection; no test hyperparameter search or switching the served model after looking at test accuracy.

## Actual held-out results

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Served PyTorch MLP | 95.27% | 0.9528 |
| SVM comparison | 97.89% | 0.9788 |

The SVM performs better. The served model is deliberately and explicitly the PyTorch network for neural-inference demonstration, not a claim that deep learning is always superior. Accuracy is for this historical benchmark only, not arbitrary handwriting/photos or general OCR. No uncertainty calibration, rotation robustness or present-day production guarantee.

![MLP confusion matrix](reports/confusion_matrix.png)

See exact metrics, hashes, split indices and all held-out predictions in `reports/`. Tests independently reload the PyTorch checkpoint and reproduce **every** held-out class prediction, check source hashes, split separation, recompute metrics, exercise API routes and reject invalid input.

## Deployment status

Local browser/HTTP checking is recorded in `reports/browser_check.json`. Dockerfile and GitHub CI are preparations, not claimed successful container/cloud runs. Docker commands: `docker build -t digit-recognition-lab .` then `docker run --rm -p 8000:8000 digit-recognition-lab`. No cloud hosting, paid service, production authentication or monitoring is implied. Public deployment requires account permission, cost/security review and independent live checks.

## Sources, license and provenance

Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49

Official record and CC BY 4.0 statement: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Original archive: https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip

License: https://creativecommons.org/licenses/by/4.0/legalcode . Original `optdigits.tra`, `optdigits.tes`, `optdigits.names` redistributed unchanged with attribution; no endorsement implied. Model code/output and fixed normalization are project additions. Downloaded 2026-11-14; hashes recorded in metrics.

Initial code/docs/training/tests prepared with an AI assistant on 2026-11-14. Not proof of the applicant's prior independent work or expertise. See `CONTRIBUTIONS.md` and `EXPLAIN_AR.md`. The applicant should understand, run and adapt the work and state their real contribution and AI assistance.
