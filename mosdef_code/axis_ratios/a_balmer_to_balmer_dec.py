
def convert_attenuation_to_dec(A_Balmer):
    """Converts balmer attenuation to Balmer dec
    
    Parameters:
    A_Balmer(float): Balmer attenuation
    
    """
    R_V = 3.1
    balmer_dec = 2.86*(10**(A_Balmer/(R_V*2.32)))
    return balmer_dec

# print(convert_attenuation_to_dec(10))