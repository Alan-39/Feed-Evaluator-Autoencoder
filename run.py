import argparse

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train", nargs=3, help="Trains a new model from dataset. Takes 3 args -t img_path gt_path model_name")
    parser.add_argument("-p", "--predict_single", nargs=2, help="Loads existing model to predict on image. Takes 2 args -p model_name img_path")
    parser.add_argument("-b", "--predict_batch", nargs=2, help="*Work in progress* Loads existing model to predict on directory of images. Takes 2 args -p model_name img_dir")
    parser.add_argument("-pv", "--predict_video", nargs=2, help="Predict on a video. -pv model_path video_path")
    parser.add_argument("-s", "--segment_background", nargs=2, help="")   
    parser.add_argument("-c", "--labelme2dataset", nargs=1, help="Convert COCO JSON dataset to image and gt-dots. -cc json_path")
    parser.add_argument("-g", "--gtmask2gtdot", nargs=1, help="Convert GT mask to GT dots. -cd gt-path")
    parser.add_argument("-sh", "--show_history", nargs=1, help="Show model's training history. -sh history_json_path")
    
    args = parser.parse_args()

    # model_1 - epoch 100, batch_size = 4, optimizer='adadelta', autoencoder
    # model_2 - epoch 100, batch_size = 4, optimizer='adadelta', unet
    # model_3 - epoch 200, batch_size = 4, optimizer='adadelta', autoencoder
    # model_4 - epoch 200, batch_size = 4, optimizer='adadelta', unet

    if (args.train):
        from model import train_model
        args = args.train
        train_model(args[0], args[1], args[2])
    if (args.predict_single):
        from model import predict_image
        args = args.predict_single
        predict_image(args[0], args[1])
    if (args.predict_batch): # WIP, not fully functional
        from model import predict_batch
        args = args.predict_batch
        predict_batch(args[0], args[1])
    if (args.predict_video):
        from model import predict_video
        args = args.predict_video
        predict_video(args[0], args[1])
    if (args.segment_background):
        from utilities import segment_background
        args = args.segment_background
        segment_background(args[0], args[1])
    if (args.labelme2dataset):
        from utilities import labelmejson_to_dataset
        args = args.labelme2dataset
        labelmejson_to_dataset(args[0])
    if (args.gtmask2gtdot):
        from utilities import gtmask_to_gtdots
        args = args.gtmask2gtdot
        gtmask_to_gtdots(args[0])
    if (args.show_history):
        from utilities import show_training_history
        import json
        args = args.show_history
        with open(args[0]) as f:
            data = json.load(f)
        show_training_history(data)