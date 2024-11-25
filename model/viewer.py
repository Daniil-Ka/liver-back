from dicom2jpg import io2img
from PIL import Image
import io

# Загрузка DICOM-файла
dicom_file_path = r"C:\Users\dkrap\Desktop\liver-back\model\i0034,0000b.dcm"

# Read and convert the DICOM file to an image
with open(dicom_file_path, 'rb') as dicom_file:
    dicom_content = dicom_file.read()

# Convert DICOM content to a numpy image
image_data = io2img(io.BytesIO(dicom_content))

# Convert the numpy image to a PIL Image
image = Image.fromarray(image_data)

# Save the image as JPEG to a BytesIO stream for export
output_path = "/mnt/data/converted_image.jpg"
image.save(output_path, format="JPEG")

