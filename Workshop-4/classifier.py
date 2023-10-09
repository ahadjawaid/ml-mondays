from fastai.vision.all import *
import gradio as gr

class Classifier:
    def __init__(self, model_path):
        self.learn = load_learner(model_path)

    def predict(self, image):
        pred, pred_idx, probs = self.learn.predict(image)
        return {self.learn.dls.vocab[i]: float(p) for i, p in enumerate(probs)}
    
image = gr.inputs.Image(shape=(128, 128))
label = gr.outputs.Label()
classifer = Classifier(model_path="model.pkl")

interface = gr.Interface(fn=classifer.predict, inputs=image, outputs=label)
interface.launch(inline=False)