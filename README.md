Forked to try to validate and improve transkun

steps: 
1) prepare data: 
- maestro_dataset v3: https://magenta.withgoogle.com/datasets/maestro#dataset
store in: 
- /scratch/gilbreth/li5042/datasets/maestro_dataset

2) replicate results: re-train transkun

3) data augmentation: 
        ["Dataset Name", "Server Location", "Instrument", "Audio Type", "Count"],
- train/validation
 - ["MSMD", "/depot/yunglu/data/transcription/msmd_data", "Piano", "wav", "467"],
 - ["BiMMuDa", "/depot/yunglu/data/transcription/BiMMuDa", "Piano", "wav", "375"],
- ["POP909", "/depot/yunglu/data/transcription/POP909", "Piano", "wav", "909"],

retrain with all

4) test: re-verify with test dataset. 


# Fixes

## Bugs in transkun

train.py: 
dropout_last -> drop_last