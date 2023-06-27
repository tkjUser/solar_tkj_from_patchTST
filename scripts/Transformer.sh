if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Transformer

root_path_name=dataset/solar_Australasian
data_path_name=data_36_resample2_51984.csv
model_id_name=Transformer
data_name=custom

random_seed=2023
for pred_len in 96   # 192 336 720
do
    python -u run_solar.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 6 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 2048 \
      --dropout 0.1\
      --fc_dropout 0.2\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done