import sys
sys.path.insert(0, "/home/grads/tianjiao/tunix")
from tunix.models.paligemma import model as pali
import inspect
fns = [n for n, obj in inspect.getmembers(pali.PaLIGemmaConfig) if isinstance(obj, classmethod)]
print("Available PaLIGemmaConfig factories:\n", "\n ".join(fns))
