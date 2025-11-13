import drms

c = drms.Client(email='oliver.e.rice@durham.ac.uk')
#Find the correct Carrington Rotation for this date.

seg = c.query(('mdi.synoptic_mr_polfil_96m[%4.4i]' % 2000), seg='Br_polfil')

crot_times = c.query(('mdi.synoptic_mr_polfil_96m'), key = ["T_START","T_STOP","CAR_ROT"])
