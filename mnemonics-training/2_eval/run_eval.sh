python main.py --nb_cl_fg=50 --nb_cl=2 --nb_protos=20 --epochs=160 --gpu=0 --dataset=cifar100 --random_seed=1993 --use_mtl
python main.py --nb_cl_fg=50 --nb_cl=2 --nb_protos=20 --epochs=90 --gpu=0 --dataset=imagenet_sub --data_dir=./data/seed_1993_subset_100_imagenet/data --num_workers=16 --test_batch_size=50 --use_mtl
python main.py --nb_cl_fg=500 --nb_cl=20 --nb_protos=20 --epochs=90 --gpu=0 --dataset=imagenet --data_dir=./data/imagenet/data --num_workers=16 --test_batch_size=50 --num_classes=1000 --use_mtl
