
def check_for_nan(data):
    # Verificar la presencia de NaN en el DataFrame
    has_nan = data.isna().any().any()
    return has_nan