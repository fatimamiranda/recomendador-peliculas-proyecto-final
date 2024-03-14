import os

def save_to_excel(movies):
    module_dir = os.path.dirname(__file__)
    excel_file_path = os.path.join(module_dir, 'excel.xlsx')
    
    if not os.path.exists(excel_file_path):
        movies.to_excel(excel_file_path, index=True)