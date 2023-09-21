# Documentation for `starspot` utils

## `rotation_tools.py`
Functions for manipulating stellar light curves.

::: starspot.rotation_tools
    handler: python
    selection:
      members:
        - transit_mask
        - load_kepler_data
        - load_and_normalize
        - dan_acf
        - interp
        - simple_acf
        - find_and_mask_transits
        - apply_masks
        - butter_bandpass_filter
        - get_peak_statistics
        - filter_sigma_clip
        - sigma_clip
    rendering:
      show_root_heading: true
      show_source: false

## `sigma_clipping.py`
Functions for sigma clipping.
::: starspot.sigma_clipping
    handler: python
    selection:
      members:
        - filter_sigma_clip
        - sigma_clip
    rendering:
      show_root_heading: true
      show_source: false

## `phase_dispersion_minimization.py`

Functions for the phase dispersion minimization period estimator.

::: starspot.phase_dispersion_minimization
    handler: python
    selection:
      members:
        - sj2
        - s2
        - calc_phase
        - phase_bins
        - phi
        - gaussian
        - nll
        - estimate_uncertainty
        - fit_gaussian
    rendering:
      show_root_heading: true
      show_source: false