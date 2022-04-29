import cv2
import json


def draw_bbox(image, bbox_list, color):
    for bbox in bbox_list:
        center = (round(bbox['center'][0]), round(bbox['center'][1]))
        label = bbox['label']
        left = round(bbox['left'])
        top = round(bbox['top'])
        right = round(bbox['left']) + round(bbox['width'])
        bottom = round(bbox['top']) + round(bbox['height'])
        img_h, img_w, _ = image.shape
        thick = round((img_h + img_w) // 900)
        cv2.rectangle(image, (left, top), (right, bottom), color, thick)
        cv2.putText(image, label, (left, top - 4), 0, 5e-4 * img_h, color, thick // 3)
        cv2.circle(image, center, radius=1, color=color, thickness=-1)


def main():
    img = cv2.imread('./output/3d_scene/train/images/3d_train_000017.png')
    a = 0
    with open('./output/3d_scene/train/scenes_3d_train.json', 'r', encoding='utf-8') as f:
        json_dic = json.load(f)
    scene = json_dic['scenes'][a]['objects']
    scene_p = json_dic['scenes'][a]['planes']

    bbox_list = []
    for i in range(len(scene)):
        bbox_list.append(
            {
                "left": scene[i]['bounding_box']['left'],
                "top": scene[i]['bounding_box']['top'],
                "width": scene[i]['bounding_box']['width'],
                "height": scene[i]['bounding_box']['height'],
                "label": 'o'+str(i),
                "center": scene[i]['pixel_coords']
            }
        )
    draw_bbox(img, bbox_list, (0, 0, 255))


    bbox_list = []
    for i in range(len(scene_p)):
        bbox_list.append(
            {
                "left": scene_p[i]['bounding_box']['left'],
                "top": scene_p[i]['bounding_box']['top'],
                "width": scene_p[i]['bounding_box']['width'],
                "height": scene_p[i]['bounding_box']['height'],
                "label": 'p'+str(i),
                "center": scene_p[i]['center_pixel_coords']
            }
        )

    draw_bbox(img, bbox_list, (255, 0, 0))
    cv2.imwrite('./bbox.png', img)
    # for p in range(len(json_dic['scenes'][a]["planes"][:-1])):
    #     center = (json_dic['scenes'][a]["planes"][p]['center_pixel_coords'][0], json_dic['scenes'][a]["planes"][p]['center_pixel_coords'][1])
    #     cv2.circle(img, center, radius=2, color=(0, 0, 255), thickness=-1)
    # cv2.imwrite("output.png", img)


main()
