from config import config
# root of image locations
images_dir = config['image_root']
# 根据图片ID获得图片的具体路径
def image_path(image_id):
    """
    Get image location from image ID
    """
    first_index = str(int(int(image_id) % 100 / 10))
    second_index = str(int(image_id) % 100)
    return ''.join([images_dir, "/Images_", first_index, "/", second_index, "/", str(image_id).strip(), ".jpg"])