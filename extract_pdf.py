from pypdf import PdfReader

reader = PdfReader('SENSORS-PAPER PDF.pdf')
text = '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())
with open('paper.txt', 'w', encoding='utf-8') as f:
    f.write(text)
