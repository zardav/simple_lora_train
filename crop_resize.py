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
        
    def resize(self, img_path):
        pred = model(img_path)[0]
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
        xy_diff = image.size[0] - image.size[1]

        
