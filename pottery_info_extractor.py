from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from scipy.optimize import linear_sum_assignment #pip
import os
import re
import io
import base64
from openai import OpenAI #pip
def encode_numpy_array_to_binary(np_array):
    # 将NumPy数组转换为Image对象
    image = Image.fromarray(np_array)
    with io.BytesIO() as output_buffer:
        image.save(output_buffer, format='PNG')
        binary_image_data = output_buffer.getvalue()
    encoded_image_data = base64.b64encode(binary_image_data).decode('utf-8')
    return encoded_image_data
HEADERS = {
    "Content-Type": "application/json",
    # "api-key": "155966a08ac9403795b726df61bf5d65"
    # "Authorization":"Bearer sk-t4cK0u91BuiLNIDbs4YuOWQ0RvSzDuoxk4CVGfPsyhWxBycq"
    "Authorization":""
}
GPT_TURBO_URL="https://api.chatanywhere.com.cn/v1/chat/completions"



from PIL import Image
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def calculate_intersection_area(box1, box2):
    # 计算两个边界框的相交面积
    # box1和box2的格式为[xc, yc, w, h]
    # 返回相交面积值
    x1_min = box1[0] - box1[2] / 2  # 计算 box1 的左上角 x 坐标
    y1_min = box1[1] - box1[3] / 2  # 计算 box1 的左上角 y 坐标
    x1_max = box1[0] + box1[2] / 2  # 计算 box1 的右下角 x 坐标
    y1_max = box1[1] + box1[3] / 2  # 计算 box1 的右下角 y 坐标
    x2_min = box2[0] - box2[2] / 2  # 计算 box2 的左上角 x 坐标
    y2_min = box2[1] - box2[3] / 2  # 计算 box2 的左上角 y 坐标
    x2_max = box2[0] + box2[2] / 2  # 计算 box2 的右下角 x 坐标
    y2_max = box2[1] + box2[3] / 2  # 计算 box2 的右下角 y 坐标
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    intersection_area = width * height
    return intersection_area
def IOU_change(coordinate:torch.Tensor,coordinate1:torch.Tensor):
    inter=calculate_intersection_area(coordinate,coordinate1)
    return inter/(coordinate[-1]*coordinate[-2])


def rectangle_overlap_area(r1, r2):
    """
    Calculate the overlapping area of two rectangles.
    Each rectangle is defined by a tuple (x_min, y_min, x_max, y_max).
    """
    x_min1, y_min1, x_max1, y_max1 = r1
    x_min2, y_min2, x_max2, y_max2 = r2

    # Calculate the overlap boundaries
    x_overlap_min = max(x_min1, x_min2)
    y_overlap_min = max(y_min1, y_min2)
    x_overlap_max = min(x_max1, x_max2)
    y_overlap_max = min(y_max1, y_max2)

    # Calculate the width and height of the overlap area
    overlap_width = max(0, x_overlap_max - x_overlap_min)
    overlap_height = max(0, y_overlap_max - y_overlap_min)

    # Calculate the overlap area
    overlap_area = overlap_width * overlap_height

    return overlap_area

def rectangle_area(r):
    """
    Calculate the area of a rectangle.
    Each rectangle is defined by a tuple (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = r
    return (x_max - x_min) * (y_max - y_min)
def bbox_to_xyxy(bbox):
    """
    Convert a bounding box from (xc, yc, w, h) format to (x_min, y_min, x_max, y_max) format.
    """
    xc, yc, w, h = bbox
    x_min = xc - w / 2
    y_min = yc - h / 2
    x_max = xc + w / 2
    y_max = yc + h / 2
    return (x_min, y_min, x_max, y_max)

def is_mostly_inside(inner_rect, outer_rect, threshold=0.8):
    """
    Determine if most of the inner rectangle is inside the outer rectangle.
    """
    overlap_area = rectangle_overlap_area(inner_rect, outer_rect)
    inner_area = rectangle_area(inner_rect)
    return (overlap_area / inner_area) > threshold

def parse_string(input_string):
    # 定义正则表达式来匹配并提取各部分内容
    pattern = r'\d+\.[\u4e00-\u9fa5]+\..+：\d'
    match = re.match(pattern, input_string)
    if match:
        return True
    else:
        return False






def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def get_cost_matrix(image_dict_this_index,text_this_index):
    cost_matrix=np.zeros((len(image_dict_this_index),len(text_this_index)))
    for i in range(len(image_dict_this_index)):
        bbox_image=np.array(tuple(image_dict_this_index[i].values()))[0][0]
        center1 = calculate_center(bbox_image)
        for j in range(len(text_this_index)):
            bbox_text=np.array(text_this_index[j]['bbox'])
            center2 = calculate_center(bbox_text)
            cost_matrix[i][j]=euclidean_distance(center1, center2)
    return cost_matrix

def assign_tasks(cost_matrix):
    num_workers, num_tasks = cost_matrix.shape

    if num_workers > num_tasks:
        # 添加虚拟任务
        dummy_costs = np.full((num_workers, num_workers - num_tasks), fill_value=np.max(cost_matrix) + 1)
        cost_matrix = np.hstack((cost_matrix, dummy_costs))
    elif num_tasks > num_workers:
        # 添加虚拟工人
        dummy_costs = np.full((num_tasks - num_workers, num_tasks), fill_value=np.max(cost_matrix) + 1)
        cost_matrix = np.vstack((cost_matrix, dummy_costs))

    # 使用 linear_sum_assignment 函数
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 过滤掉虚拟工人或虚拟任务的分配结果
    valid_indices = (row_ind < num_workers) & (col_ind < num_tasks)
    row_ind = row_ind[valid_indices]
    col_ind = col_ind[valid_indices]

    # 输出分配结果
    # print("分配结果:")
    output_pair=[]
    for i, j in zip(row_ind, col_ind):
        # print(f"工人 {i} 分配到任务 {j}，成本: {cost_matrix[i, j]}")
        output_pair.append([i,j])
    return output_pair

def convert_float32_to_float64(obj):
    if isinstance(obj, dict):
        # 如果是字典，递归地处理字典中的每个键值对
        return {key: convert_float32_to_float64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # 如果是列表，递归地处理列表中的每个元素
        return [convert_float32_to_float64(item) for item in obj]
    elif isinstance(obj, np.float32):
        # 如果是浮点数，转换为float64
        return np.float64(obj)
    else:
        # 其他类型不做处理，直接返回
        return obj

prompt_template='现在你是一个阅读理解机器人，你会阅读并深度理解我给你的文本内容并据此回答我所提出的问题。\n注意，我给出的问题是：给定文本‘图三八六QDI①层出土陶器 1、4、6、7、9.侈沿罐(QDIT1①：3、QDIT1①：2、5、QDIT1①：4、QDIT6①：6) 2.筒形罐(QDIT1①：6) ’\n这样的文本你要整理成:\n<begin>1："QDIT1①：侈沿罐"\n4："QDIT1①："侈沿罐"\n6："QDIT1①：侈沿罐"\n7："QDIT1①：侈沿罐"\n9："QDIT6①：侈沿罐\n2："QDIT1①：筒形罐"</begin>\n你需要阅读理解的文本是（提示：①简洁不要啰嗦②格式完全按照示例输出,整理好的文本一定要放在<begin></begin>中间）③示例内容不要干扰理解文本，仅关注要理解的文本内容，，尤其是冒号后面的数字））：'
# 设置常量
TEXT_PROMPT = "all objects.words"  # 文本提示
# TEXT_PROMPT ="Arabic numerals"
BOX_TRESHOLD = 0.15  # 物体框的阈值
TEXT_TRESHOLD = 0.15  # 文本区域的阈值
color_map = [
    (255, 0, 0),  # 红色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 粉红
    (0, 255, 255),  # 青色
    # 继续添加更多颜色和类别
]  # 物体框颜色
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-t4cK0u91BuiLNIDbs4YuOWQ0RvSzDuoxk4CVGfPsyhWxBycq",
    base_url="https://api.chatanywhere.tech/v1"
)

# 文件名
# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract pottery information from images')
parser.add_argument('--output_dir', type=str, 
                    help='Directory to save output files')
parser.add_argument('--image_dir', type=str,help='Directory to save input images')
parser.add_argument('--config_gd', type=str,help='Path to config file of GroundingDINO')
parser.add_argument('--pretrained_weight_gd',help='Path to pretrained weights xof GroundingDINO')
args = parser.parse_args()
model = load_model(args.config_gd,  args.pretrained_weight_gd, device='cpu')
# 构建图像路径
base_dir=args.image_dir
all_images_name=os.listdir(base_dir)#59:
annotations=[]
image_id=-1
num = 0
for s in tqdm(range(len(all_images_name))):
    name=all_images_name[s]
    error_list = []
    IMAGE_PATH = os.path.join(base_dir,name)
    # 加载原始图像和处理后的图像
    image_source, image = load_image(IMAGE_PATH)

    # 使用目标检测模型预测物体框和文本区域
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device='cpu'
    )
    # 调整物体框的坐标到图像的实际尺寸
    h_, w_, _ = image_source.shape
    boxes = boxes * torch.Tensor([w_, h_, w_, h_])
    image_source=np.array(image_source, dtype=np.int32)
    # 遍历每个物体框，绘制框并显示置信度
    i=0
    all_crop_text_list=[]
    all_crop_image_list=[]
    while i < len(boxes):
        confidence, [xc, yc, w, h] = logits[i], boxes[i]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        if phrases[i]=='objects':
            len_phases=1
        else:
            len_phases=2
        color = color_map[len_phases % 6]  # 框的颜色，这里为绿色
        thickness = 2  # 框的线宽
        start_point = (int(x1-5), int(y1))
        end_point = (int(x2+5), int(y2+20))
        if 'objects' in phrases[i]:
            count_inc = 0
            for j in range(len(boxes)):
                if i == j:
                    continue
                c_box = boxes[j]
                iou = IOU_change(c_box, boxes[i])
                c_box_area = c_box[-1] * c_box[-2]
                box_area = boxes[i][-1] * boxes[i][-2]
                if iou > 0.9:
                    # and box_area > c_box_area
                    count_inc += 1
            if count_inc >= 2:
                # or 45 in list(sorted_dict.keys())[-1:]
                boxes = torch.cat((boxes[:i], boxes[i + 1:]))
                phrases = phrases[:i] + phrases[i + 1:]
                continue
        i+=1
    i=0

    while i < len(boxes):
        is_mostly_inside_flag = False
        crop_image_index = {}
        crop_text_dict = {}
        confidence, [xc, yc, w, h] = logits[i], boxes[i]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        if phrases[i]=='objects':
            len_phases=1
            i+=1
            continue
        else:
            len_phases=2
        color = color_map[len_phases % 6]  # 框的颜色，这里为绿色
        thickness = 2  # 框的线宽
        start_point = (int(x1-5), int(y1))
        end_point = (int(x2+5), int(y2+20))
        input_box = np.array([x1, y1, x2, y2])

        # 提取物体框对应的图像区域
        crop_img = image_source[max(int(y1),0):int(y2)+80, max(int(x1)-70,0):int(x2)+80]
        crop_img = np.uint8(crop_img)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        crop_img = cv2.filter2D(crop_img, -1, kernel)
        # # 加载预训练的FSRCNN模型
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "ESPCN_x4.pb"  # 替换为你的模型路径
        # sr.readModel(path)
        # sr.setModel("espcn", 4)  # 模型名称和放大倍数
        # # 执行超分辨率
        # crop_img = sr.upsample(np.uint8(image))


        # 智慧图问展示
        if 'objects' in phrases[i]:
            # crop_img_resize=cv2.resize(crop_img, (crop_img.shape[1] // 10, crop_img.shape[0] // 10))
            encoded_image=encode_numpy_array_to_binary(crop_img)
            response = client.chat.completions.create(
                model="gpt-4.1-nano",  # 填写需要调用的模型名称
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    # "url": fileurl
                                    "url":f"data:image/jpeg;base64,{encoded_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "检测图中的数字,只输出1个数字，不要别的文字描述"
                            }
                        ]
                    }
                ]
            )
            print('提问：'+'输出图中的数字,只输出1个数字，不要别的文字描述')
            # prompt_template_2='‘OCR检测的内容如下：图三八六QDI①层出土陶器 1、4、6、7、9.侈沿罐(QDIT1①：3、QDIT1①：2、QDIT1①：5、 QDIT1①：4、QDIT6①：6)2.敞口折腹钵(QDIT3①：1)3、5.小 口双耳罐(QDIT7①：3、QDIT6①：5)8.鼓腹罐(QDIT6①：3) 10、11.平口罐(QDIT7①：2、QDIT1①：1)12.双耳罐(QDIT6①：2) ’你要把上述文字整理成：图三八六QDI①层出土陶器\n 1、4、6、7、9.侈沿罐(QDIT1①：3、QDIT1①：2、QDIT1①：5、 QDIT1①：4、QDIT6①：6)\n2.敞口折腹钵(QDIT3①：1)\n3、5.小 口双耳罐(QDIT7①：3、QDIT6①：5)\n8.鼓腹罐(QDIT6①：3) \n10、11.平口罐(QDIT7①：2、QDIT1①：1)\n12.双耳罐(QDIT6①：2)。注意 ①不要改变任何内容，只需要用‘\n’断句，②示例中图三八六QDI①层出土陶器要与后面数字断开。下面是你要整理的内容：'
            resp_str=response.choices[0].message.content
            print('回答：'+resp_str)
            numbers = re.findall(r'\d+', resp_str)
            # 将提取到的数字转换为整数
            numbers = [int(num) for num in numbers]
            if len(numbers)!=0:
                number=numbers[0]
            else:
                j = 0

                while j < len(boxes):
                    if i == j:
                        j += 1
                        continue
                    c_box = boxes[j]
                    if is_mostly_inside(bbox_to_xyxy(boxes[i]), bbox_to_xyxy(c_box)):
                        boxes = torch.cat((boxes[:i], boxes[i + 1:]))
                        phrases = phrases[:i] + phrases[i + 1:]
                        is_mostly_inside_flag = True
                        break
                    j += 1
                number = -1
            if number in crop_image_index.keys():
                crop_image_index[number].append(input_box)
            else:
                crop_image_index.update({number:[input_box]})
            all_crop_image_list.append(crop_image_index)
        if not is_mostly_inside_flag and 'words' in phrases[i]:
            # crop_img_resize=cv2.resize(crop_img, (crop_img.shape[1] // 10, crop_img.shape[0] // 10))
            encoded_image = encode_numpy_array_to_binary(crop_img)
            response = client.chat.completions.create(
                model="gpt-4o",  # 填写需要调用的模型名称
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    # "url": fileurl
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "检测图中的文字,并进行形如例子中的整理，例：1、2、7~9.侈沿罐（QDIF24：8、10、9、1 QDIF25：7） 2、6.\n小口双耳罐 (QDIF1：4、1)整理为‘1.侈沿罐.QDIF24：8\n2.侈沿罐.QDIF24：10\n7.侈沿罐.QDIF24：9\n8.侈沿罐.QDIF24：1\n9.侈沿罐.QDIF25：7\n2.小口双耳罐.QDIF1：4\n6.小口双耳罐.QDIF1：1’ 。请先输出检测到的文字，然后回答‘整理后的内容为：xxx’。若没有检测到文字则回答‘无文字’,若检测的文字不符合上述形式则回答‘形式有误’"
                            }
                        ]
                    }
                ]
            )
            print('提问：检测图中的文字,并进行形如例子中的整理，例：1、2、7~9.侈沿罐（QDIF24：8、10、9、1 QDIF25：7）5.敞口折腹钵（QDIF24：13）整理为‘1.侈沿罐.QDIF24：8\n2.侈沿罐 QDIF24：10\n7.侈沿罐.QDIF24：9\n8.侈沿罐.QDIF24：1\n9.侈沿罐.QDIF25：7\n5.敞口折腹钵 QDIF24：13’ 。请先输出检测到的文字，然后回答‘整理后的内容为：xxx’。若没有检测到文字则回答无文字,若检测的文字不符合上述形式则回答‘形式有误')
            resp_str=response.choices[0].message.content
            print('回答：'+resp_str)

            resp_str_split=resp_str.split('整理后的内容为：')[-1].split('\n')
            output_list =[]
            for split_str in resp_str_split:
                if split_str=='' or '无文字' in split_str or '形式有误' in split_str or '未指定' in split_str:
                    continue
                elif not str.isdigit(split_str[0]):
                    continue
                elif not bool(re.compile(r'[\u4e00-\u9fa5]+').search(split_str)):
                    continue
                elif not parse_string(split_str):
                    continue
                else:
                    dict_one={
                        'index': split_str.split('.')[0],
                        'class': split_str.split('.')[1],
                        'item': split_str.split('.')[2].split('：')[0],
                        'bbox': list(input_box)
                    }
                    output_list.append(dict_one)
                all_crop_text_list=all_crop_text_list+output_list
        if not is_mostly_inside_flag:
            i+=1
    aggregated_dict = {}
    # 遍历所有的字典项
    for item in all_crop_image_list:
        for key, value in item.items():
            if key not in aggregated_dict:
                aggregated_dict[key] = value
            else:
                aggregated_dict[key].extend(value)
    for key, boxes in aggregated_dict.items():
        filtered_boxes = []
        for i in range(len(boxes)):
            inside_any = False
            for j in range(len(boxes)):
                if i != j and is_mostly_inside(boxes[i], boxes[j]):
                    inside_any = True
                    break
            if not inside_any:
                filtered_boxes.append(boxes[i])
        aggregated_dict[key] = filtered_boxes
    # Deduplicate all_crop_text_list by comparing all dictionary values
    all_crop_text_unique_list = []
    seen_items = set()
    
    for item in all_crop_text_list:
        # Create a hashable representation of the item's values
        item_key = tuple((k, str(v)) for k, v in item.items())
        
        if item_key not in seen_items:
            seen_items.add(item_key)
            all_crop_text_unique_list.append(item)
    all_text_index=[]
    for list_tmp in all_crop_text_unique_list:
        index_tmp=list_tmp['index']
        all_text_index.append(index_tmp)
    all_text_index_unique_list = []
    seen = set()
    for item in all_text_index:
        if item not in seen:
            all_text_index_unique_list.append(item)
            seen.add(item)
    num=0

    for i in range(len(all_text_index_unique_list)):
        index_str=all_text_index_unique_list[i]
        index_digit=int(all_text_index_unique_list[i])
        image_dict_index_tmp=[]
        if index_digit in aggregated_dict.keys():
            for tmp in aggregated_dict[index_digit]:
                image_dict_index_tmp.append({index_digit:[tmp]})
        # tuple(all_crop_image_list[0].values())[0][0]
        indices = [index for index, value in enumerate(all_text_index) if value == index_str]
        text_this_index = [all_crop_text_unique_list[j] for j in indices]
        if len(image_dict_index_tmp)==0:
            error_list.append('图片'+IMAGE_PATH+' '+str(text_this_index[0]['index']) + '号器出现问题！')
            print('图片'+IMAGE_PATH+' '+str(text_this_index[0]['index']) + '号器出现问题！')
            continue
        matrix=get_cost_matrix(image_dict_index_tmp,text_this_index)
        pair_list=assign_tasks(matrix)
        if len(image_dict_index_tmp)!=len(text_this_index):
            potential_error_message='图片'+IMAGE_PATH+' '+str(list(image_dict_index_tmp[pair_list[0][0]].keys())[0]) + '号器可能出现问题！'
            print(potential_error_message)
            error_list.append(potential_error_message)

        for pair in pair_list:
            bbox=list(image_dict_index_tmp[pair[0]].values())[0][0]
            crop_img_tmp=image_source[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            info=text_this_index[pair[1]]
            class_name=info['class']
            item=info['item']
            path=os.path.join(args.output_dir,os.path.basename(IMAGE_PATH).split('.')[0],item,class_name)
            image_name=class_name+'_'+item+'_'+str(num)+'.png'

            # 如果目录不存在，则创建
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imencode('.png', crop_img_tmp)[1].tofile(os.path.join(path,image_name))
            num+=1


    # Create error file path using the command line argument
    error_path = os.path.join(args.output_dir,os.path.basename(IMAGE_PATH).split('.')[0], "error_list_" + os.path.basename(IMAGE_PATH).split('.')[0] + '.txt')
    # Create directory for error file if it doesn't exist
    error_dir = os.path.dirname(error_path)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    # 打开文件并写入错误信息
    with open(error_path, 'w', encoding='utf-8') as file:
        for error in error_list:
            file.write(error + '\n')
    print('完成图片'+os.path.basename(IMAGE_PATH)+'的提取')
