try:
    from vggface2.interface import InterfaceVggface2
except ImportError:
    from feature_extract.vggface2.interface import InterfaceVggface2


class ExtractFeature:
    def __init__(self, option):
        if not hasattr(ExtractFeature, 'model_name') or option != self.model_name:
            self.model_name = option
            if option in ["resnet50_256_pytorch", "resnet50_128_pytorch", "resnet50_ft_pytorch",
                            "resnet50_scratch_pytorch", "senet50_scratch_pytorch",
                            "senet50_ft_pytorch", "senet50_256_pytorch", "senet50_128_pytorch"]:
                self.interface_obj = InterfaceVggface2(option)

    def get_feature_file(self, path_image):
        return self.interface_obj.extract_file(path_image)

    def get_feature_folder(self, path_folder):
        return self.interface_obj.extract_folder(path_folder)
