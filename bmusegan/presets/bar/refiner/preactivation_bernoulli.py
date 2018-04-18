"""Refiner built with residual blocks and bernoulli activation
"""
NET_R = {}

NET_R['private'] = [
    ('reshape', (96, 84, 1)),                              
    ('identity', None, None, 'bn', 'lrelu'),
    ('conv2d', (128, (3, 12), (1, 1), 'SAME'), 'bn', 'lrelu'),  
    ('conv2d', (1, (3, 12), (1, 1), 'SAME')),    
    ('identity', None, None, 'bernoulli', ('add', 0)),
    ('reshape', (1, 96, 84, 1))
]
