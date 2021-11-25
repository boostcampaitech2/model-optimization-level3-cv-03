"""Module generator related to pooling operations.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
# pylint: disable=useless-super-delegation
from torch import nn

from src.modules.base_generator import GeneratorAbstract


class MaxPool1d(nn.Module):
    def __init__(self,kernel_size,stride):
        super(MaxPool1d, self).__init__()
        self.maxpool1d = nn.MaxPool1d(kernel_size=kernel_size,stride=stride);

    def forward(self,x):
        result = self.maxpool1d(x.permute(0,2,1));    
        return result.permute(0,2,1);
    

class MaxPool1dGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channel

    def __call__(self, repeat: int = 1):
        module = (
            [MaxPool1d(*self.args) for _ in range(repeat)]
            if repeat > 1
            else MaxPool1d(*self.args)
        )
        return self._get_module(module)
        
        

class MaxPoolGenerator(GeneratorAbstract):
    """Max pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channel

    @property
    def base_module(self) -> nn.Module:
        """Base module."""
        return getattr(nn, f"{self.name}2d")

    def __call__(self, repeat: int = 1):
        module = (
            [self.base_module(*self.args) for _ in range(repeat)]
            if repeat > 1
            else self.base_module(*self.args)
        )
        return self._get_module(module)


class AvgPoolGenerator(MaxPoolGenerator):
    """Average pooling module generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    """Global average pooling module."""

    def __init__(self, output_size=1):
        """Initialize."""
        super().__init__(output_size=output_size)


class GlobalAvgPoolGenerator(GeneratorAbstract):
    """Global average pooling module generator."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__(*args, **kwargs)
        self.output_size = 1
        if len(args) > 1:
            self.output_size = args[1]

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        return self._get_module(GlobalAvgPool(self.output_size))
