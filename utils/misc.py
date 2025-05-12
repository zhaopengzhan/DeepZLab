'''
“misc”是“miscellaneous”的缩写，意为“杂项”或“其他”
放点工具直接导入就行

'''
import datetime
import functools
import time

import torch


def calRunTime(func):
    """ 计算函数运行时间的装饰器 """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"函数 `{func.__name__}` 运行时间: {end_time - start_time:.6f} 秒")
        return result

    return wrapper


class calRunTimer():
    def __init__(self, func=None, block_name=''):
        self.func = func
        self.block_name = block_name
        if func is not None:
            # 如果作为装饰器使用
            # 保留函数的属性（如__name__, __doc__等）
            functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 `{self.func.__name__}` 运行时间: {end_time - start_time:.4f} 秒")
        return result

    def __enter__(self):
        """ 上下文管理器计算运行时间 """
        self.start_time = time.time()  # 记录开始时间
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 上下文管理器计算运行时间 """
        self.end_time = time.time()  # 记录结束时间
        elapsed_time = self.end_time - self.start_time
        print(f"代码块`{self.block_name}`运行时间 运行时间: {elapsed_time:.4f} 秒")


class Wrapper:
    '''
    自己写的封装类，封装Loss\AverageMeter
    '''

    def __init__(self):
        self._container = {}

    def register(self, name, value):
        """通过名字添加或更新损失函数"""
        self._container[name] = value

    def __getattr__(self, name):
        """尝试访问不存在的属性时返回 None"""
        return self._container.get(name, None)

    def __getitem__(self, item):
        """尝试访问不存在的属性时返回 None"""
        return self._container.get(item, None)

    def keys(self):
        return self._container.keys()


class Result:
    '''
    GroupVIT里面搞得结果封装类
    '''

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


class AverageMeter:
    # 计算平均MAE、MSE等平均指标
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        """通过名字添加或更新损失函数"""
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


def truncate_tensor(tensor, target_shape):
    """截断 tensor 以匹配 target_shape"""
    slices = tuple(slice(0, min(tensor.size(i), target_shape[i])) for i in range(len(target_shape)))
    return tensor[slices]

def pad_tensor(tensor, target_shape):
    """补充 tensor 以匹配 target_shape
    现在是0填充
    """
    pad_width = []
    for i in range(len(target_shape)):
        diff = target_shape[i] - tensor.size(i)
        pad_width.extend([0, max(0, diff)])  # 在每一维的末尾补充
    result = torch.nn.functional.pad(tensor, pad=pad_width[::-1])
    return result

def adjust_checkpoint(checkpoint, model_state):

    updated_checkpoint = {}

    for key, pretrained_weight in checkpoint.items():
        if key in model_state:
            model_weight = model_state[key]
            if pretrained_weight.shape != model_weight.shape:
                print(f"Resizing {key}: {pretrained_weight.shape} -> {model_weight.shape}")
                pretrained_weight = truncate_tensor(pretrained_weight, model_weight.shape)
                # 先搞截断，再搞填充
                if pretrained_weight.shape != model_weight.shape:
                    pretrained_weight = pad_tensor(pretrained_weight, model_weight.shape)
            updated_checkpoint[key] = pretrained_weight
        else:
            print(f"Skipping {key}, not found in model.")

    return updated_checkpoint


if __name__ == '__main__':
    loss_wrapper = Wrapper()
    loss_wrapper.register('loss_hr', AverageMeter())
    loss_wrapper.register('loss_lr', AverageMeter())
    loss_wrapper.register('loss_sum', AverageMeter())

    loss_wrapper.loss_hr.register(1, 1)
    loss_wrapper.loss_hr.register(2, 3)

    print(loss_wrapper.loss_mr)
