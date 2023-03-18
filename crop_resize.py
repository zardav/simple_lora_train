import yolov7
from PIL import Image


def ensure_coco_label_available():
    try:
        import coco_labels
    except ImportError:
        from subprocess import run
        run('wget https://github.com/kadirnar/yolov7-pip/raw/77641b757d695b4840f25d7844b3507cd6b80cff/yolov7/deploy/triton-inference-server/labels.py -O coco_label.py', shell=True)

        
def n_to_label(n):
    ensure_coco_label_available()
    from coco_labels import COCOLabels
    return str(COCOLabels(n)).split('.')[1].lower()
   
    
class Resizer:
    def __init__(self):
        self.model = yolov7.load('yolov7.pt')
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.classes = None  # (optional list) filter by class
        
    def get_square_image(self, img_path):
        x1, y1, x2, y2 = self.get_square(img_path)
        image = Image.open(img_path)
        width, height = image.size
        result = Image.new(image.mode, (x2-x1, y2-y1), 'black')
        to_paste = image.crop((max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)))
        result.paste(to_paste, (-x1, -y1))
        return result
        
    def get_square_image_of_size(self, img_path, size):
        return self.get_square_image(img_path).resize((size, size))
        
    def get_square(self, img_path):
        pred = self.model(img_path)[0]
        person_indices = [n_to_label(x)=='person' for x in pred[:, 5]]
        filtered_pred = pred[person_indices, :]
        boxes = filtered_pred[:, :4] # x1, y1, x2, y2
        scores = filtered_pred[:, 4]
        min_x = filtered_pred[:, 0].min()
        min_y = filtered_pred[:, 1].min()
        max_x = filtered_pred[:, 2].max()
        max_y = filtered_pred[:, 3].max()
        margin = 20
        minimum_box = min_x-margin, min_y-margin, max_x+margin, max_y+margin
        image = Image.open(img_path)
        width, height = image.size
        xy_diff = width - height
        if xy_diff > 0:
            # First: symmetric crop
            new_x1 = min(0+xy_diff//2, minimum_box[0])
            new_x2 = max(width-xy_diff//2, minimum_box[2])
            new_width = new_x2-new_x1
            if new_width == height:
                return new_x1, 0, new_x2, height
            # Then perform max crop
            cropped_right = width - new_x2
            new_x1 = min(0+xy_diff-cropped_right, minimum_box[0])
            cropped_left = new_x1
            new_x2 = max(width-(xy_diff-cropped_left), minimum_box[2])
            new_width = new_x2 - new_x1
            if new_width == height:
                return new_x1, 0, new_x2, height
            # Then symmetric pad
            y_to_pad = new_width - height
            return new_x1, -y_to_pad//2, new_x2, height+y_to_pd//2
        elif xy_diff < 0:
            yx_diff = -xy_diff
            # First: symmetric crop
            new_y1 = min(0+yx_diff//2, minimum_box[1])
            new_y2 = max(height-xy_diff//2, minimum_box[3])
            new_height = new_y2-new_y1
            if new_height == width:
                return 0, new_y1, width, new_y2
            # Then perform max crop
            cropped_bottom = height - new_y2
            new_y1 = min(0+yx_diff-cropped_bottom, minimum_box[1])
            cropped_top = new_y1
            new_y2 = max(height-(yx_diff-cropped_top), minimum_box[3])
            new_height = new_y2 - new_y1
            if new_height == width:
                return 0, new_y1, width, new_y2
            # Then symmetric pad
            y_to_pad = new_width - height
            return new_x1, -y_to_pad//2, new_x2, height+y_to_pd//2
        else:
            return 0, 0, width, height
        
