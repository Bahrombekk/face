import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

def align_face(image, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    left_eye_center = np.array(left_eye, dtype=np.float32)
    right_eye_center = np.array(right_eye, dtype=np.float32)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    desired_right_eye_x = 1.0 - 0.35
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - 0.35) * 256
    scale = desired_dist / dist
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                   (left_eye_center[1] + right_eye_center[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = 256 * 0.6
    tY = 256 * 0.43
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    (w, h) = (256, 256)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return output

def process_image(image_path, output_folder):
    try:
        detector = MTCNN()
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Xato: {image_path} rasmni o'qib bo'lmadi."
        
        result = detector.detect_faces(image)
        if result:
            face = result[0]
            landmarks = face['keypoints']
            aligned_face = align_face(image, landmarks)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            
            if os.path.exists(output_path):
                return True, f"Qayta ishlash shart emas: {output_path} allaqachon mavjud."
            
            cv2.imwrite(output_path, aligned_face)
            return True, f"Saved aligned face to {output_path}"
        else:
            return False, f"Yuz aniqlanmadi: {image_path}"
    except Exception as e:
        return False, f"Xatolik yuz berdi {image_path} bilan ishlashda: {str(e)}"

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_filenames = set(os.listdir(output_folder))
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename not in processed_filenames
    ]
    
    results = []
    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(process_image, path, output_folder) for path in image_paths]
        for future in future_results:
            results.append(future.result())
    
    return results

def run_alignment(input_folder, output_folder="datasets"):
    results = process_images(input_folder, output_folder)
    success_count = sum(1 for success, _ in results if success)
    total_count = len(results)
    
    return {
        "success": success_count > 0,
        "message": f"{success_count}/{total_count} rasmlar muvaffaqiyatli qayta ishlandi",
        "details": [msg for _, msg in results]
    }