
 ## Installation

 ```bash
 git clone https://github.com/yourname/yolo-rcnn-pipeline.git
 cd yolo-rcnn-pipeline
-pip install -r requirements.txt       # OR  pip install -e .[wandb,kaggle,hf]
+# install **editable** so the `yrcnn` command & imports are created
+pip install -e .[wandb,kaggle,hf]     # add extras as you wish
bash +yrcnn viz coco --n 2 +pytest -q +


