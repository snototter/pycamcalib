"""Filters for the preprocessing pipeline."""

# import everything is done to ensure each implemented filter is registered
# and can thus be created via :func:`pcc.preproc.filters.create_filter()`.

from pcc.preproc.filters.filter_base import *
from pcc.preproc.filters.color_correction import *
from pcc.preproc.filters.equalization import *
from pcc.preproc.filters.thresholding import *
