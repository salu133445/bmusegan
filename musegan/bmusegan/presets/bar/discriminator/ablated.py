"""Network architecture of the ablated (without onset and chroma streams)
discriminator
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
    ('conv2d', (64, (1, 1), (1, 1)), None, 'lrelu'),    # 0  (16, 84)
]

NET_D['shared'] = [
    ('conv2d', (128, (4, 3), (4, 2)), None, 'lrelu'),   # 0  (4, 3)
    ('conv2d', (256, (1, 3), (1, 3)), None, 'lrelu'),   # 1  (4, 1)
]

NET_D['onset'] = None

NET_D['chroma'] = None

NET_D['merged'] = [
    ('conv2d', (512, (1, 1), (1, 1)), None, 'lrelu'),   # 0  (4, 1)
    ('reshape', (4*512)),
    ('dense', 1),
]
