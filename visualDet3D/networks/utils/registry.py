import inspect
class Registry(object):
    
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def __getitem__(self, key):
        return self.module_dict[key]

    def _register_module(self, module_class, force=False):
        """Register a module.
        Args:
            module : Module to be registered.
        """
        if (not inspect.isclass(module_class)) and (not inspect.isfunction(module_class)):
            raise TypeError('module must be a class or function, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None):
        self._register_module(cls)
        return cls

DATASET_DICT  = Registry("datasets")
BACKBONE_DICT = Registry("backbones")
DETECTOR_DICT = Registry("detectors")
PIPELINE_DICT = Registry("pipelines")
AUGMENTATION_DICT = Registry("augmentation")
SAMPLER_DICT = Registry("sampler")
