"""Network architecture of the proposed discriminator
"""
NET_D = {}

NET_D['pitch_time_private'] = [
    ('reshape', (96, 84, 1)),                           # 0  (96, 84)
    ('conv2d', (32, (1, 12), (1, 12)), None, 'lrelu'),  # 1  (96, 7)
    ('conv2d', (64, (6, 1), (6, 1)), None, 'lrelu'),    # 2  (16, 7)
]

NET_D['time_pitch_private'] = [
    ('reshape', (96, 84, 1)),                           # 0  (96, 84)
    ('conv2d', (32, (6, 1), (6, 1)), None, 'lrelu'),    # 1  (16, 84)
    ('conv2d', (64, (1, 12), (1, 12)), None, 'lrelu'),  # 2  (16, 7)
]

NET_D['merged_private'] = [
    ('conv2d', (64, (1, 1), (1, 1)), None, 'lrelu'),    # 0  (16, 7)
]

NET_D['shared'] = [
    ('conv2d', (128, (4, 3), (4, 2)), None, 'lrelu'),   # 0  (4, 3)
    ('conv2d', (256, (1, 3), (1, 3)), None, 'lrelu'),   # 1  (4, 1)
]

NET_D['onset'] = [
    ('reshape', (96, 84, 13)),                          # 0  (96, 84)
    ('sum', (2), True),                                 # 1  (96, 1)
    ('conv2d', (64, (6, 1), (6, 1)), None, 'lrelu'),    # 2  (16, 1)
    ('conv2d', (128, (4, 1), (4, 1)), None, 'lrelu'),   # 3  (4, 1)
]

NET_D['chroma'] = [
    ('conv2d', (128, (1, 12), (1, 12)), None, 'lrelu'), # 0  (4, 1)
]

NET_D['merged'] = [
    ('conv2d', (512, (1, 1), (1, 1)), None, 'lrelu'),   # 0  (4, 1)
    ('reshape', (4*512)),
    ('dense', 1),
]
