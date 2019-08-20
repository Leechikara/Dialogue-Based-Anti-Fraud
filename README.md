# Dialogue-Based-Anti-Fraud
Code and data for EMNLP-IJCNLP 2019 paper "Are You for Real? Detecting Identity Fraud via Dialogue Interactions".

## Requirements
```
python 3.6
pytorch >= 0.4.1
matplotlib
numpy
```

## Train models
If you just want to pre-train the dialogue models, use
```
cd src
./warm_up.sh
```

If you want to pre-train the dialogue models and use reinforcement learning, use
```
cd src
./rl.sh
```

## Test models
```
cd src
./warm_up.sh
```

## Notes
- Please read the annotation in the scripts. 
- When testing, please load the correct trained models. 
- When training HP-S (--model_setting hrl), please use bonus mechanism to stable training (just a trick). Otherwise, the rl training will collapse.
- Contact me if you have any questions.
