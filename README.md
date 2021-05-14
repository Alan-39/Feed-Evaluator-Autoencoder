This project attempt to optimise the initial feed evaluator implementation by creating a new lightweight package from ground up without the use of a cumbersome external image segmentation library.

==================== 10/5/2021 ====================<br>
Implementation has been overhauled. Adding assorted functions used to convert dataset from coco, find centroid, etc, as well as separation of preprocessor and model functions for easier code management


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

Convert COCO JSON dataset to img and gt-mask dataset
```
py run.py -cc "json_path"
```

Convert gt-mask to gt-dots
```
py run.py -cd "gt_path"
```

Show trained model history
```
py run.py -sh "history_json_path"
```
