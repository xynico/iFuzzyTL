# iFuzzyTL
code for iFuzzyTL: Interpretable Fuzzy Transfer Learning for SSVEP BCI System ( https://doi.org/10.48550/arXiv.2410.12267)

'''
    model para = dict{
        seq_len = 150, -> Data Length
        embed_dim = 32, -> # Channels
        dropout = 0.3, -> dropout
        num_classes = n_class, -> num_classes in classifier head
        classifier_direction = 's2c', #s2c -> the order of the dimension
        
        encoder_module_para = dict( -> Encoder para. Here only show the FuzzyDualAttention 
            name = 'FuzzyDualAttention', -> Name
            num_rules=10, -> # Rules
            num_heads = [1,1], -> # head for attention
            softmax = 'log_softmax', -> log softmax or softmax
            layer_sort = ['s', 'e'], -> order of the Fuzzy filters
            use_projection=True, -> Whether use query projector. In our test, no-proj did not work at all.
            norm=True, -> Whether add norm
            methods = ['l2', 'l2'], -> How to calculate distance. 
        ),
    
        # The length N of num_heads, layer_sort, and methods should be same. N -> number of Fuzzy filters.
        }
'''
