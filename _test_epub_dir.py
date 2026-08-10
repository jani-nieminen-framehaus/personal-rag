import tempfile
from pathlib import Path
import zipfile

def build_epub_v2(path, chapters, title='Test Book'):
    """EPUB 2.0.1 format - more widely compatible."""
    with zipfile.ZipFile(str(path), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        zf.writestr('META-INF/container.xml', '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        manifest_items = '\n'.join('    <item id="%s" href="%s.xhtml" media-type="application/xhtml+xml"/>' % (c['id'], c['id']) for c in chapters)
        manifest_items += '\n    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        spine_items = '\n'.join('    <itemref idref="%s"/>' % c['id'] for c in chapters)
        opf = '<?xml version="1.0" encoding="UTF-8"?><package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>%s</dc:title><dc:language>en</dc:language><dc:identifier id="uid">urn:uuid:test-book-001</dc:identifier></metadata><manifest>%s</manifest><spine toc="ncx">%s</spine></package>' % (title, manifest_items, spine_items)
        zf.writestr('OEBPS/content.opf', opf)
        nav_points = ''
        for i, c in enumerate(chapters):
            nav_points += '<navPoint id="np%d" playOrder="%d"><navLabel><text>%s</text></navLabel><content src="%s.xhtml"/></navPoint>' % (i+1, i+1, c['title'], c['id'])
        ncx = '<?xml version="1.0" encoding="UTF-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head><meta name="dtb:uid" content="urn:uuid:test-book-001"/></head><docTitle><text>%s</text></docTitle><navMap>%s</navMap></ncx>' % (title, nav_points)
        zf.writestr('OEBPS/toc.ncx', ncx)
        for c in chapters:
            body_escaped = c['body'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            xhtml = '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>%s</title></head><body><h1>%s</h1><p>%s</p></body></html>' % (c['title'], c['title'], body_escaped)
            zf.writestr('OEBPS/%s.xhtml' % c['id'], xhtml)

import sys
sys.path.insert(0, 'D:/Tinkering sideprojects/rag')
from ebooklib import epub as ep

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    epub = tmp / 'test.epub'
    build_epub_v2(epub, [{'id': 'ch1', 'title': 'Chapter One', 'body': 'Story.'}])
    print('EPUB created at:', epub)
    try:
        book = ep.read_epub(str(epub))
        items = list(book.get_items())
        print('Items:', len(items))
        for item in items:
            print(' ', item.file_name, item.get_type())
        print('Spine:', book.spine)
    except Exception as e:
        print('ERROR:', e)
        import traceback
        traceback.print_exc()
