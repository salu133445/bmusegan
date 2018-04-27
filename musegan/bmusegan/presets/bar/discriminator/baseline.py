"""Network architecture of the baseline discriminator
"""
NET_D = {}

NET_D['pitch_time_private'] = None

NET_D['time_pitch_private'] = None

NET_D['merged_private'] = None

NET_D['shared'] = None

NET_D['onset'] = None

NET_D['chroma'] = None

NET_D['merged'] = [
    ('reshape', (96, 84, 13)),                              # 0  (96, 84)
    ('conv2d', (128, (1, 12), (1, 12)), None, 'lrelu'),     # 1  (96, 7)
    ('conv2d', (128, (1, 3), (1, 2)), None, 'lrelu'),       # 2  (96, 3)
    ('conv2d', (256, (6, 1), (6, 1)), None, 'lrelu'),       # 3  (16, 3)
    ('conv2d', (256, (4, 1), (4, 1)), None, 'lrelu'),       # 4  (4, 3)
    ('conv2d', (512, (1, 3), (1, 3)), None, 'lrelu'),       # 5  (4, 1)
    ('reshape', (4*512)),
    ('dense', 1),
]
