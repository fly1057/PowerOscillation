import fitz
import os
import glob


def jpg_png2pdf(path='.'):
    for name in glob.glob(os.path.join(path, '*.png')):
        imgdoc = fitz.open(name)
        pdfbytes = imgdoc.convertToPDF()    # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        imgpdf.save(name[:-4] + '.pdf')

    for name in glob.glob(os.path.join(path, '*.jpg')):
        imgdoc = fitz.open(name)
        pdfbytes = imgdoc.convertToPDF()    # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        imgpdf.save(name[:-4] + '.pdf')


jpg_png2pdf()
