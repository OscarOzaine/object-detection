import torch.cuda
import yolov5.train as yolo_train
import os

def train_yolo(**kwargs):
    params = get_yolo_params(**kwargs)
    #print(**params)
    yolo_train.run(**params)


def get_yolo_params(name, epochs=50, project='models', weights='yolov5s',
                    evolve=0, device='cuda:0'):
    yolo_parameters = {
        'data': '/home/oscar/Projects/object-detection/SolarPanel/src/models/yolov5/sp_dataset.yaml',
        # 'weights': f'{project}/{weights}/weights/best.pt',
        'weights': weights + '.pt' if not weights.endswith('.pt') else '',
        'imgsz': 256,
        # 'batch_size': 16,
        'batch_size': 48,
        'workers': 4,
        'project': project,
        'name': name,
        'epochs': epochs,
        'device': device,
    }
    if evolve:
        yolo_parameters['evolve'] = evolve
        yolo_parameters['name'] += '_hyp'
    return yolo_parameters

def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0, cuda:1, cuda:2"
    device = 'cuda:0, cuda:1, cuda:2' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    model = 'yolov5s'
    model_name = f'{model}_{epochs}_test'

    train_yolo(
        name=model_name,
        epochs=epochs,
        project='trained_models',  
        #evolve=10,
        weights=model, 
        device=device
    )


if __name__ == '__main__':
    main()
