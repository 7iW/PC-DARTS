
#No diretorio acima ao train_data_path é necessario ter um labels.txt
python train_search_big_paral.py  --image_size 512  --data_path '../dataset/FIGURAS_ML_PUC_2019_Co/' --labels_path '../src/classification/train_png.txt' --epochs 300 --n_class 5 --batch_size 64
