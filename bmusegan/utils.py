"""Project-specific utility functions
"""
import numpy as np
import imageio
from bmusegan.lib.image_grid import get_image_grid
from bmusegan.lib.midi_io import write_midi

def get_num_parameter(var_list):
    """Given the variable list, return the total number of parameters."""
    return int(np.sum([np.product([x.value for x in var.get_shape()])
                       for var in var_list]))

def save_image(filepath, phrases, shape, inverted=True, grid_width=3,
               grid_color=0, frame=True):
    """
    Save a batch of phrases to a single image grid.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : np.array, ndim=5
        The phrase array. Shape is (num_phrase, num_bar, num_time_step,
        num_pitch, num_track).
    shape : list or tuple of int
        Shape of the image grid. (height, width)
    inverted : bool
        True to invert the colors. Default to True.
    grid_width : int
        Width of the grid lines. Default to 3.
    grid_color : int
        Color of the grid lines. Available values are 0 (black) to
        255 (white). Default to 0.
    frame : bool
        True to add frame. Default to True.
    """
    if phrases.dtype == np.bool_:
        if inverted:
            phrases = np.logical_not(phrases)
        clipped = (phrases * 255).astype(np.uint8)
    else:
        if inverted:
            phrases = 1. - phrases
        clipped = (phrases * 255.).clip(0, 255).astype(np.uint8)

    flipped = np.flip(clipped, 3)
    transposed = flipped.transpose(0, 4, 1, 3, 2)
    reshaped = transposed.reshape(-1, phrases.shape[1] * phrases.shape[4],
                                  phrases.shape[3], phrases.shape[2])

    merged_phrases = []
    phrase_shape = (phrases.shape[4], phrases.shape[1])
    for phrase in reshaped:
        merged_phrases.append(get_image_grid(phrase, phrase_shape, 1,
                                             grid_color))

    merged = get_image_grid(np.stack(merged_phrases), shape, grid_width,
                            grid_color, frame)
    imageio.imwrite(filepath, merged)

def save_midi(filepath, phrases, config):
    """
    Save a batch of phrases to a single MIDI file.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : list of np.array
        Phrase arrays to be saved. All arrays must have the same shape.
    pause : int
        Length of pauses (in timestep) to be inserted between phrases.
        Default to 0.
    """
    if phrases.dtype != np.bool_:
        phrases = (phrases > 0.5)
    reshaped = phrases.reshape(-1, phrases.shape[1] * phrases.shape[2],
                               phrases.shape[3], phrases.shape[4])
    pad_width = ((0, 0), (0, config['pause_between_samples']),
                 (config['lowest_pitch'],
                  128 - config['lowest_pitch'] - config['num_pitch']),
                 (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant')
    transposed = padded.reshape(-1, padded.shape[2], padded.shape[3]).transpose(
        2, 0, 1)
    write_midi(filepath, transposed, config['programs'], config['is_drums'],
               tempo=config['tempo'])
