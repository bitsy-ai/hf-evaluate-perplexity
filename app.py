import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("grepLeigh/perplexity")
launch_gradio_widget(module)