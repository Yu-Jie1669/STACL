python trainer.py \
  --input \
  data/train/train.zh.bpe.min \
  data/train/train.en.bpe.min \
  --vocabulary \
  data/nist.20k.zh.vocab \
  data/nist.10k.en.vocab \
  --model \
  transformer \
  --validation \
  data/val/val.zh.bpe \
  --references \
  data/val/val.en.\* \
  --process_group \
  nccl \
  --parameters=batch_size=16,update_cycle=2,device_list=[1] \
  --hparam_set \
  base
