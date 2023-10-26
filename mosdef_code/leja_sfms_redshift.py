

def leja2022_sfms(logM, z, type='ridge'):
    """Star-forming main sequence parameterized model
    From https://iopscience.iop.org/article/10.3847/1538-4357/ac887d/pdf
    
    Parameters:
    logM (float): log stellar mass
    z (float): redshift
    type (str): Either 'ridge' or 'mean' 
    """
    if type=='ridge':
        a = 0.03746 + 0.3448*z + (-0.1156*z**2)
        b = 0.9605 + 0.04990*z + (-0.05984*z**2)
        c = 0.2516 + 1.118*z + (-0.2006*z**2)
        logMt = 10.22 + 0.3826*z + (-0.04491*z**2)

    if type=='mean':
        a = -0.06707 + 0.3684*z + (-0.1047*z**2)
        b = 0.8552 + (-0.1010*z) + (-0.001816*z**2)
        c = 0.2148 + 0.8137*z + (-0.08052*z**2)
        logMt = 10.29 + (-0.1284*z) + (0.1203*z**2)
    
    # if type=='ridge':
    #     a = -0.2384 + 1.204*z + (-0.5929*z**2)
    #     b = 0.9387 + 0.005499*z + (-0.02751*z**2)
    #     c = 0.3257 + 0.8805*z + (-0.06114*z**2)
    #     logMt = 10.37 + 0.06952*z + (0.1252*z**2)

    # if type=='mean':
    #     a = 0.04849 + 0.1386*z + (-0.004984*z**2)
    #     b = 0.9224 + (-0.1613*z) + (0.01574*z**2)
    #     c = 0.1372 + 0.9583*z + (-0.1442*z**2)
    #     logMt = 10.15 + (0.1215*z) + (0.009843*z**2)

    if logM > logMt:
        logSFR = a*(logM-logMt) + c
    if logM <= logMt:
        logSFR = b*(logM-logMt) + c
    
    return logSFR
        
