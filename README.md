# RADTTS + iSTFTNet vocoder

Install dependencies:

```bash
pip install -r requirements.txt
```

Download Ukrainian RADTTS and iSTFTNet models:

```bash
mkdir models
cd models

wget https://github.com/egorsmkv/ukrainian-radtts/releases/download/v1.0/RADTTS-Lada.pt
wget https://github.com/egorsmkv/ukrainian-radtts/releases/download/v1.0/iSTFTNet-Vocoder-Lada.pt
```

Then you can inference own texts by the following command:

```bash
python3 inference.py -c config_ljs_dap.json -r models/RADTTS-Lada.pt -t test_sentences.txt --vocoder_checkpoint_file models/iSTFTNet-Vocoder-Lada.pt -o results/
```