import easyocr
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


reader = easyocr.Reader(['en', 'th'])
results = reader.readtext('invoice.jpg')

text = ''

for result in results:
    if result[1] == 'address629/1':
        text += result[1] + ''

print("text", text)