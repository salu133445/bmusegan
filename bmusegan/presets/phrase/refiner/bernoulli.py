"""Refiner built with residual blocks and bernoulli activation
"""
NET_R = {}

NET_R['private'] = [
    ('identity', None, None, None),
    ('conv3d', (64, (1, 1, 12), (1, 1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (32, (1, 3, 1), (1, 1, 1), 'SAME'), 'bn', 'relu'),  
    ('conv3d', (1, (1, 3, 12), (1, 1, 1), 'SAME'), 'bn', None),
    ('identity', None, None, 'relu', ('add', 0)),
    ('conv3d', (64, (1, 1, 12), (1, 1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (32, (1, 3, 1), (1, 1, 1), 'SAME'), 'bn', 'relu'),   
    ('conv3d', (1, (1, 3, 12), (1, 1, 1), 'SAME'), 'bn', None),    
    ('identity', None, None, 'bernoulli', ('add', 4)),
]
