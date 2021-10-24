mkdir -p ./generated_params
mkdir -p ./logs
CUDA_VISIBLE_DEVICES=0 python search.py --config config/PEMS_BAY_para.yaml --epoch 60 |& tee logs/search_PEMS_BAY.log
CUDA_VISIBLE_DEVICES=0 python train.py --config config/PEMS_BAY_para.yaml --epoch 100 |& tee logs/train_PEMS_BAY.log
CUDA_VISIBLE_DEVICES=0 python search.py --config config/METR_LA_para.yaml --epoch 60 |& tee logs/search_METR_LA.log
CUDA_VISIBLE_DEVICES=0 python train.py --config config/METR_LA_para.yaml --epoch 100 |& tee logs/train_METR_LA.log

CUDA_VISIBLE_DEVICES=0 python test.py --config config/PEMS_BAY_para.yaml |& tee logs/test_PEMS_BAY.log
CUDA_VISIBLE_DEVICES=0 python test.py --config config/METR_LA_para.yaml |& tee logs/test_METR_LA.log