python mtl_CartoonGAN.py --name mtl_hayao_2_paprika --src_data src --tgt_data paprika --vgg_model pretrained_model/vgg19-dcbb9e9d.pth --batch_size 15 --G_pre_trained_weight pretrained_model/Hayao_net_G_float.pth --lrD 5e-4 --lrG 1e-3 --train_epoch 300 --con_lambda 1

python mtl_CartoonGAN.py --name mtl_hayao_2_name --src_data src --tgt_data name --vgg_model pretrained_model/vgg19-dcbb9e9d.pth --batch_size 15 --G_pre_trained_weight pretrained_model/Hayao_net_G_float.pth --lrD 5e-4 --lrG 1e-3 --train_epoch 300 --con_lambda 1
