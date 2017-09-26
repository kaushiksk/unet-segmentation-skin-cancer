from tf_unet import unet, util, image_util
data_provider = image_util.ImageDataProvider("./train/*",data_suffix=".jpg",mask_suffix="_Segmentation.png")

output_path = "./model_val/"
#setup & training
net = unet.Unet(layers=3, features_root=16, channels=3, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=32, epochs=100,restore=True)