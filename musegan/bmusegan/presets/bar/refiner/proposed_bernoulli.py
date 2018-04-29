"""Refiner built with residual blocks and bernoulli activation
"""
NET_R = {}

NET_R['private'] = [
    ('identity', None, None, None),
    ('identity', None, 'bn', 'relu'),
    ('conv3d', (64, (3, 12), (1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (1, (3, 12), (1, 1), 'SAME'), None, None),
    ('identity', None, None, None, ('add', 0)),
    ('identity', None, 'bn', 'relu'),
    ('conv3d', (64, (3, 12), (1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (1, (3, 12), (1, 1), 'SAME'), None, None),
    ('identity', None, None, 'bernoulli', ('add', 4)),
    ('reshape', (1, 96, 84, 1))
]
