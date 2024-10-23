# iFuzzyTL

**Code for "iFuzzyTL: Interpretable Fuzzy Transfer Learning for SSVEP BCI System"**

Paper: [https://doi.org/10.48550/arXiv.2410.12267](https://doi.org/10.48550/arXiv.2410.12267)

This repository contains the implementation details for the iFuzzyTL model. Below are the key parameters and configurations:

```python
model_params = {
    'seq_len': 150,  # Length of data sequence
    'embed_dim': 32,  # Number of channels
    'dropout': 0.3,  # Dropout rate
    'num_classes': n_class,  # Number of classes in classifier head
    'classifier_direction': 's2c',  # Dimension order for classifier

    'encoder_module_params': {
        'name': 'FuzzyDualAttention',  # Encoder name
        'num_rules': 10,  # Number of fuzzy rules
        'num_heads': [1, 1],  # Number of attention heads
        'softmax': 'log_softmax',  # Type of softmax used
        'layer_sort': ['s', 'e'],  # Order of fuzzy filters
        'use_projection': True,  # Whether to use a query projector
        'norm': True,  # Whether to apply normalization
        'methods': ['l2', 'l2'],  # Distance calculation methods
    },

    # Ensure the length of 'num_heads', 'layer_sort', and 'methods' arrays are the same.
}
