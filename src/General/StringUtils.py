def preface_char(string:str, length:int, char:str='0'):
    '''
    Pads the string with the given character at the front to make it of the given length.
    '''
    l = len(string)

    if len(char) != 1:
        raise ValueError('Input char must be a single character! Input was of length ' + str(len(char)) + '.')

    if l > length:
        print('WARNING: (preface_char): Input string is longer than length! Input was ' + string +
              ', length was ' + str(length) + '.')
        return string
    
    result = ''
    for _ in range(length-l):
        result += char

    return result + string


def postface_char(string:str, length:int, char:str='0'):
    '''
    Pads the string with the given character at the end to make it of the given length.
    '''
    l = len(string)

    if len(char) != 1:
        raise ValueError('Input char must be a single character! Input was of length ' + str(len(char)) + '.')

    if l > length:
        print('WARNING: (preface_char): Input string is longer than length! Input was ' + string +
              ', length was ' + str(length) + '.')
        return string
    
    result = str(string)
    for _ in range(length-l):
        result += char

    return result


def add_trailing_zeros(string:str, n_decimals):
    '''
    Pads the string with zeros at the end to make it have the given number of decimals.
    '''

    result = str(string)

    if not '.' in string:
        raise ValueError("The given string has no decimal point!")
    
    for _ in range(n_decimals - len(string.split('.')[-1])):
        result += '0'

    return result
