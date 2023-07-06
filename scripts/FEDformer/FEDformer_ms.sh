export CUDA_VISIBLE_DEVICES=1

#cd ..

python -u run_solar.py \
  --is_training 1 \
  --root_path ./dataset/solar_Australasian/ \
  --data_path data_36_resample2_51984.csv \
  --model_id solar_FEDformer_Fourier_MS_96_96 \
  --version Fourier\
  --model FEDformer \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --gpu 0



# ETTh1
python -u run_solar.py \
  --is_training 1 \
  --root_path ./dataset/solar_Australasian/ \
  --data_path data_36_resample2_51984.csv \
  --model_id solar_FEDformer_Wavelets_MS_96_96 \
  --model FEDformer \
  --version Wavelets\
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --gpu 0

