"""Refiner built with residual blocks and binary round activation
"""
NET_R = {}

NET_R['private'] = [
    ('reshape', (96, 84, 1)),                                   # 0  (96, 84)
    ('conv2d', (128, (3, 12), (1, 1), 'SAME'), 'bn', 'lrelu'),  # 1  (96, 84)
    ('conv2d', (1, (3, 12), (1, 1), 'SAME'), 'bn', 'lrelu'),    # 2  (96, 84)
    ('identity', None, None, 'round', ('add', 0)),
    ('reshape', (1, 96, 84, 1))
]
