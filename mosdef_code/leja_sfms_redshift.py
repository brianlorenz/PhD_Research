

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
    
    if logM > logMt:
        logSFR = a*(logM-logMt) + c
    if logM <= logMt:
        logSFR = b*(logM-logMt) + c
    
    return logSFR
        
