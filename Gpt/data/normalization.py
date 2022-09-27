"""
Simple normalization functions for neural network parameters.
"""


def get_normalizer(normalizer_name, **kwargs):
    return _normalizer_mapping[normalizer_name](**kwargs)


__all__ = [get_normalizer]


class OpenAINormalization:

    """
    Follows the normalization scheme of CLIP latents used by the first stage of DALL-E 2.
    In order to compute the openai_coeff, you can use the included data/prepare_checkpoints.py script.
    """

    def __init__(self, openai_coeff, **ignore_kwargs):
        self.openai_coeff = openai_coeff

    def normalize(self, x):
        x = x * self.openai_coeff
        return x

    def unnormalize(self, x):
        x = x / self.openai_coeff
        return x

    def get_range(self, min_val, max_val):
        norm_min_val = min_val * self.openai_coeff
        norm_max_val = max_val * self.openai_coeff
        return norm_min_val, norm_max_val

    def message(self):
        return f"OpenAI(openai_coeff={self.openai_coeff})"


class IdentityNormalization:

    """
    Identity normalization (leaves parameters unchanged).
    """

    def __init__(self, **ignore_kwargs):
        pass

    @staticmethod
    def normalize(x):
        return x

    @staticmethod
    def unnormalize(x):
        return x

    @staticmethod
    def get_range(min_val, max_val):
        return min_val, max_val

    @staticmethod
    def message():
        return "Identity"


_normalizer_mapping = {
    "openai": OpenAINormalization,
    "none": IdentityNormalization,
}
