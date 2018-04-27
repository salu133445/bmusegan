"""Network architecture of the proposed generator.
"""
NET_G = {}

NET_G['shared'] = [
    ('dense', (12*512), 'bn', 'relu'),                      # 0
    ('reshape', (4, 3, 512)),                               # 1  (4, 3)
    ('transconv2d', (256, (4, 3), (4, 2)), 'bn', 'relu')    # 2  (16, 7)
]

NET_G['pitch_time_private'] = [
    ('transconv2d', (64, (1, 12), (1, 12)), 'bn', 'relu'),  # 0  (16, 84)
    ('transconv2d', (32, (6, 1), (6, 1)), 'bn', 'relu')     # 1  (96, 84)
]

NET_G['time_pitch_private'] = [
    ('transconv2d', (64, (6, 1), (6, 1)), 'bn', 'relu'),    # 0  (96, 7)
    ('transconv2d', (32, (1, 12), (1, 12)), 'bn', 'relu')   # 1  (96, 84)
]

NET_G['merged_private'] = [
    ('transconv2d', (1, (1, 1), (1, 1)), 'bn', 'sigmoid'),  # 0  (96, 84)
    ('reshape', (1, 96, 84, 1))
]
