import drms

c = drms.Client()
#Find the correct Carrington Rotation for this date.
crot_times = c.query(('mdi.synoptic_mr_polfil_96m'), key = ["T_START","T_STOP","CAR_ROT"])

print(crot_times)
