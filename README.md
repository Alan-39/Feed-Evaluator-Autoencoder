This project attempt to optimise the initial feed evaluator implementation by creating a new lightweight package from ground up without the use of a cumbersome external image segmentation library.
<br>
<h1>Saved trained models</h1>
<h2>model_1</h2>
epoch 100, batch_size = 4, optimizer='adadelta', autoencoder<br>
<h2>model_2</h2>
epoch 100, batch_size = 4, optimizer='adadelta', unet<br>
<h2>model_3 (current best)</h2>
epoch 200, batch_size = 4, optimizer='adadelta', autoencoder<br>
<h2>model_4</h2>
epoch 200, batch_size = 4, optimizer='adadelta', unet<br>
<br>

<h1>Commands</h1>
To see all commands
```
py run.py -h
```

Train a new model
```
py run.py -t "img_path" "gt_path" "model_name"
``` 

Predict on single image
```
py run.py -p "model_path.h5" "img_path"
```

Predict on batch of images (Work in progress)
```
py run.py -b "model_path.h5" "img_dir"
```

Predict on live video
```
py run.py -pv "model_path.h5" "video_path"
```

Segment background from image (Non-functional. Workin progress)
```
py run.py -s "img_path" "mask_path"
```


Convert labelme JSON dataset to img and gt-mask dataset
```
py run.py -c "json_path"
```

Convert gt-mask to gt-dots
```
py run.py -g "gt_path"
```

Show trained model history
```
py run.py -sh "history_json_path"
```

<h2>Predict on single image example</h2>
<p align="center">
  <img src="https://github.com/AlanSleeps/Feed-Evaluator-Autoencoder/blob/main/single_predict_example.jpg">
</p>
<h2>Predict on video example</h2>
<p align="center">
  <img src="https://github.com/AlanSleeps/Feed-Evaluator-Autoencoder/blob/main/video_predict_example.jpg">
</p>
