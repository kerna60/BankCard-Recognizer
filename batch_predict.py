"""
BankCard-Recognizer/batch_predict.py
实现银行卡图片批量识别功能
"""
import os
import cv2
import csv
from tqdm import tqdm
from east.predict import predict_txt
from crnn.predict import single_recognition
from gui.utils import hard_coords, selected_box

def batch_recognize(input_dir, output_file, east_model, crnn_model):
    """
    :param input_dir: 图片目录路径
    :param output_file: 结果保存路径(.csv/.txt)
    :param east_model: EAST模型路径
    :param crnn_model: CRNN模型路径
    """
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Filename', 'Card Number'])

        for img_file in tqdm(img_files, desc='Processing'):
            img_path = os.path.join(input_dir, img_file)
            try:
                # EAST文本定位[doc_1]
                boxes = predict_txt(img_path, east_model)
                if not boxes:
                    raise ValueError("No text region detected")

                # 坐标处理[doc_1]
                x0, y0, x1, y1 = hard_coords(boxes[0])
                img_array = cv2.imread(img_path)
                roi = selected_box(img_array, x0, y0, x1, y1)

                # CRNN识别[doc_1]
                result = single_recognition(roi, (256, 32), crnn_model)

                writer.writerow([img_file, result])
            except Exception as e:
                writer.writerow([img_file, f"ERROR: {str(e)}"])