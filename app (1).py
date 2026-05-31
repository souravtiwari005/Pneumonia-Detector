import gradio as gr
import fastai.vision.all as fai
import torch
import torchvision.transforms as T

learn = fai.load_learner('pneumonia_model_v2.pkl')
labels = learn.dls.vocab

tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    img = img.convert('RGB')
    tensor = tfms(img).unsqueeze(0)
    with torch.no_grad():
        out = learn.model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    return {str(labels[i]): float(probs[i]) for i in range(len(labels))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=2),
    title="🫁 Pneumonia Detector",
    description="Upload a chest X-ray to detect pneumonia using ResNet50. Trained on 5,000+ X-ray images."
).launch()
