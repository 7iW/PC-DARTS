
#No diretorio acima ao train_data_path é necessario ter um labels.txt
python train_search_big_paral.py  --image_size 256  --data_path 'data/' --labels_path 'labels.txt' --epochs 300 --n_class 5 --batch_size 16
