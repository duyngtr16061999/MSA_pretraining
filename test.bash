CUDA_VISIBLE_DEVICES=$1 python pretrain.py \
    --taskMatched --taskMaskLM --taskMaskV --taskMaskA --taskMatchedV --taskMatchedA \
    --fromScratch \
    --batchSize 2 --optim bert --lr 1e-4 --epochs 10