"""Network architecture of the proposed generator.
"""
NET_G = {}

NET_G['shared'] = [
    ('dense', 512, 'bn', 'relu'),                               # 0
    ('reshape', (1, 1, 512)),                                   # 1  (1, 1)
    ('transconv3d', (256, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'), # 3  (4, 1)
    ('transconv3d', (256, (1, 1, 3), (1, 1, 3)), 'bn', 'relu'), # 4  (4, 3)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'), # 5  (16, 3)
    ('transconv3d', (128, (1, 1, 3), (1, 1, 2)), 'bn', 'relu'), # 6  (16, 7)
]

NET_G['pitch_time_private'] = [
    ('transconv2d', (64, (1, 12), (1, 12)), 'bn', 'relu'),      # 0  (16, 84)
    ('transconv2d', (32, (6, 1), (6, 1)), 'bn', 'relu')         # 1  (96, 84)
]

NET_G['time_pitch_private'] = [
    ('transconv2d', (64, (6, 1), (6, 1)), 'bn', 'relu'),        # 0  (96, 7)
    ('transconv2d', (32, (1, 12), (1, 12)), 'bn', 'relu'),      # 1  (96, 84)
]

NET_G['merged_private'] = [
    ('transconv2d', (1, (1, 1), (1, 1)), 'bn', 'sigmoid'),      # 0  (96, 84)
    ('reshape', (1, 96, 84, 1)),
]
