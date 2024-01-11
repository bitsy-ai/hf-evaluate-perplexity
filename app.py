import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("./perplexity.py")
launch_gradio_widget(module)
