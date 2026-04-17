@echo off
cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"
python -m luthi.train_vision --image_dir "C:\Users\Hasha Smokes\Desktop\train2017" --annotations "E:\data\coco\annotations\captions_train2017.json" --val_image_dir "E:\data\coco\val2017" --val_annotations "E:\data\coco\annotations\captions_val2017.json" --resume "E:\runs\vision\checkpoint.luthi" --checkpoint_password "Smokew33d26" --epochs 102 --batch_size 4 --num_workers 0
pause
