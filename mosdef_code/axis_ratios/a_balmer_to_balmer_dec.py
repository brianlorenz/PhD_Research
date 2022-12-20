
def convert_attenuation_to_dec(A_Balmer):
    """Converts balmer attenuation to Balmer dec
    
    Parameters:
    A_Balmer(float): Balmer attenuation
    
    """
    balmer_dec = 2.86*(10**(A_Balmer/(4.05*1.97)))
    return balmer_dec

print(convert_attenuation_to_dec(10))