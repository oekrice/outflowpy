"""
This file contains the code to download a specific Carrington rotation. 
For now I'll use the resampling method of pfsspy, although Anthony's one is superior so I'll switch to that later.
"""
import outflowpy

def download_hmi_crot(crot_number):
    r"""
    Downloads (without email etc.) the HMI data matching the rotation number above

    Parameters
    ----------
    crot_number : int
        Carrington rotation number

    Returns
    -------
    files : :class:`jsoc_files`

    Notes
    -----
    Must be in the allowable range of numbers (post-2010 ish).
    Outputs a list of THREE files -- the requested rotation and those either side of it. 
    Left comes first, then right. So the order is [crot_number, crot_number+1, crot_number-1]
    """




