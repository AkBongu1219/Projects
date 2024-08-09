import streamlit as st
from PIL import Image
import numpy as np
from scipy.spatial import distance
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import os

# Damage Detection Model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (damage)
cfg.MODEL.WEIGHTS = os.path.join('C:/Users/Guest_User/Downloads/car_damage.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = 'cpu'  # cpu or cuda
damage_predictor = DefaultPredictor(cfg)

# Damage Parts Detection Model
cfg_mul = get_cfg()
cfg_mul.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_mul.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has five classes (headlamp, hood, rear_bumper, front_bumper, door)
cfg_mul.MODEL.WEIGHTS = os.path.join('C:/Users/Guest_User/Downloads/car_parts.pth')
cfg_mul.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg_mul.MODEL.DEVICE = 'cpu'  # cpu or cuda
part_predictor = DefaultPredictor(cfg_mul)

#Detects damaged car part
def detect_damage_part(damage_dict, parts_dict):
    max_distance = 10e9
    max_distance_dict = dict(zip(damage_dict.keys(), [max_distance] * len(damage_dict)))
    part_name = dict(zip(damage_dict.keys(), [''] * len(damage_dict)))

    for y in parts_dict.keys():
        for x in damage_dict.keys():
            dis = distance.euclidean(damage_dict[x], parts_dict[y])
            if dis < max_distance_dict[x]:
                part_name[x] = y.rsplit('_', 1)[0]

    return list(set(part_name.values()))

damage_class_map = {0: 'damage'}
parts_class_map = {0: 'Headlamp', 1: 'Rear Bumper', 2: 'Door', 3: 'Hood', 4: 'Front Bumper'}

# Streamlit UI
st.title("Car Damage and Parts Detection")

uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
   
    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)
    st.write("")

    if st.button('Predict'):
        # Damage detection
        damage_outputs = damage_predictor(image)
        damage_v = Visualizer(image[:, :, ::-1],
                              scale=0.5)
        damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

        # Parts detection
        parts_outputs = part_predictor(image)
        parts_v = Visualizer(image[:, :, ::-1],
                             scale=0.5,
                             instance_mode=ColorMode.IMAGE_BW)
        parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))
       
        # Convert the output images to a format that Streamlit can display
        damage_image = damage_out.get_image()[:, :, ::-1]
       
        # Display the images with Streamlit
        st.image(damage_image, caption='Damage Detection', use_column_width=True)
       
        # Calculate the damage and parts information
        damage_prediction_classes = [damage_class_map[el] + "_" + str(indx) for indx, el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
        damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
        damage_dict = dict(zip(damage_prediction_classes, damage_polygon_centers))

        parts_prediction_classes = [parts_class_map[el] + "_" + str(indx) for indx, el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
        parts_polygon_centers = parts_outputs["instances"].pred_boxes.get_centers().tolist()

        # Remove centers which lie beyond 800 units
        parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
        parts_dict = dict(zip(parts_prediction_classes, parts_polygon_centers_filtered))

        # Determine the damaged parts
        damaged_parts = detect_damage_part(damage_dict, parts_dict)

        # Display the damaged parts information
        if damaged_parts:
            st.write("Damaged Parts: ", ", ".join(damaged_parts))
        else:
            st.write("No damaged parts detected.")
