"""Refiner built with residual blocks and binary round activation
"""
NET_R = {}

NET_R['private'] = [
    ('identity', None, None, None),
    ('identity', None, 'bn', 'relu'),
    ('conv2d', (64, (3, 12), (1, 1), 'SAME'), 'bn', 'relu'),
    ('conv2d', (1, (3, 12), (1, 1), 'SAME'), None, None),
    ('identity', None, None, None, ('add', 0)),
    ('identity', None, 'bn', 'relu'),
    ('conv2d', (64, (3, 12), (1, 1), 'SAME'), 'bn', 'relu'),
    ('conv2d', (1, (3, 12), (1, 1), 'SAME'), None, None),
    ('identity', None, None, 'round', ('add', 4)),
    ('reshape', (1, 96, 84, 1)),
]
