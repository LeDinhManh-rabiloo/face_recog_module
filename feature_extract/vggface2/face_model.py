import sys
from os.path import join, dirname

path = join(dirname(__file__), "models")
sys.path.insert(0, path)

def get_model(name, is_eval):
    if name == "senet50_128_pytorch":
        from pytorch.senet50_128_pytorch import senet50_128 as model
        network = model.senet50_128(weights_path=join(dirname(__file__),
                                                      "models/pytorch/senet50_128_pytorch/senet50_128.pth"))
        if is_eval:
            print("senet50_128_eval")
            network.eval()
        return network
    return None
