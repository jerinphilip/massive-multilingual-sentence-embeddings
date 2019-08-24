from tqdm import tqdm
from .logging import prettified

progress_handler = {}
def register_progress_handler(tag):
    def _inner(f):
        progress_handler[tag] = f
        return f
    return _inner


class BaseProgressHandler:
    def __init__(self, iterable, progress_dict):
        self.iterable = iterable
        self.progress_dict = progress_dict

    def __iter__(self):
        return self

    def __next__(self):
        self._update()
        return next(self.iterable)

    def _update(self):
        raise NotImplementedError()



@register_progress_handler('tqdm')
class Tqdm(BaseProgressHandler):
    def __init__(self, iterable, progress_dict, **kwargs):
        defaults = {"ascii": "#", "leave": False, 
                "ncols": 200, "dynamic_ncols": True, 
                "mininterval": 30}
        defaults.update(kwargs)
        self.pbar = tqdm(iterable, postfix=None, **defaults)
        super().__init__(iter(self.pbar), progress_dict)

    def _update(self):
        self.pbar.set_postfix(self.progress_dict)
        pass

@register_progress_handler('none')
class _None(BaseProgressHandler):
    def __init__(self, iterable, *args, **kwargs):
        super().__init__(iterable, None)

    def _update(self):
        pass


@register_progress_handler('text')
class _UpdateEvery(BaseProgressHandler):
    def __init__(self, iterable, progress_dict, update_interval=30, **kwargs):
        super().__init__(iterable, progress_dict)
        self.update_interval = update_interval
        self.updates = 0
        self.kwargs = kwargs

    def _update(self):
        self.updates = self.updates + 1
        flag = (self.updates)%self.update_interval

        items = self.progress_dict.items()
        items = list(items)
        items.append(
            ( 'updates', 
            '{}/{}'.format(self.updates, self.kwargs['total']))
        )
        items = sorted(items, key=lambda x: x[0])

        if not flag:
            print(self.kwargs['desc'], prettified(items))

progress_methods = list(progress_handler.keys())
