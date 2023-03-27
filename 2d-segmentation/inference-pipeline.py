from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from utils.numpy_helper import segmentation3d
import mmcv
import glob2
import numpy as np
import os


def visualize_plot(model, img, result, out_file):
    img = model.show_result(img, result, palette=model.PALETTE, show=False, opacity=0.5)
    if out_file is not None:
        mmcv.imwrite(img, out_file)


def main():
    configs = "models/pspnet_r101-d8_480x480_80k_pascal_context_59.py"
    model_path = "models/pspnet_r101-d8_480x480_80k_pascal_context_59_20210416_114418-fa6caaa2.pth"
    model = init_segmentor(configs, model_path, device="cpu")
    model = revert_sync_batchnorm(model)
    input_image_ls = glob2.glob("input/*")
    for image in input_image_ls:
        result = inference_segmentor(model, image)
        output_path = os.path.join("output", image.split("/")[-1])
        # seg3d = segmentation3d(result[0])
        with open(output_path.split(".")[0] + ".npy", "wb") as f:
            np.save(f, result[0])
        f.close()
        visualize_plot(model, image, result, output_path)

        dictionary = {}
        flat = result[0].reshape(
            result[0].shape[0] * result[0].shape[1],
        )
        for item in flat:
            if item in dictionary.keys():
                dictionary[item] += 1
            else:
                dictionary[item] = 1
        mapper = [
            "aeroplane",
            "bag",
            "bed",
            "bedclothes",
            "bench",
            "bicycle",
            "bird",
            "boat",
            "book",
            "bottle",
            "building",
            "bus",
            "cabinet",
            "car",
            "cat",
            "ceiling",
            "chair",
            "cloth",
            "computer",
            "cow",
            "cup",
            "curtain",
            "dog",
            "door",
            "fence",
            "floor",
            "flower",
            "food",
            "grass",
            "ground",
            "horse",
            "keyboard",
            "light",
            "motorbike",
            "mountain",
            "mouse",
            "person",
            "plate",
            "platform",
            "pottedplant",
            "road",
            "rock",
            "sheep",
            "shelves",
            "sidewalk",
            "sign",
            "sky",
            "snow",
            "sofa",
            "table",
            "track",
            "train",
            "tree",
            "truck",
            "tvmonitor",
            "wall",
            "water",
            "window",
            "wood",
        ]

        with open(output_path.split(".")[0] + ".txt", "w") as t:
            t.write("key, num_pixel, class_name")
            t.write("\n")
            for key, val in dictionary.items():
                t.write(str(key) + "," + str(val) + "," + str(mapper[key]))
                t.write("\n")

        print("finished processing ----> ", image)


if __name__ == "__main__":
    main()
