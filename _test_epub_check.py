import zipfile, io
import tempfile
from pathlib import Path
sys.path.insert(0, 'D:/Tinkering sideprojects/rag')
from tests.test_epub_ingest import _build_epub

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    epub = tmp / "book.epub"
    _build_epub(epub, [{"id": "ch1", "title": "Chapter One", "body": "First chapter content."}])
    
    # Try to open
    from ebooklib import epub as epub_module
    import ebooklib
    try:
        book = epub_module.read_epub(str(epub))
        items = list(book.get_items())
        print("Opened OK, items:", len(items))
        for item in items:
            print("  type=%s  is_chapter=%s  file_name=%s  title=%s" % (
                item.get_type(), item.is_chapter(), item.file_name, item.title))
        print("Spine:", book.spine)
    except Exception as e:
        print("ERROR:", e)
        
    # Check the zip contents
    with zipfile.ZipFile(str(epub)) as zf:
        print("\nZIP contents:")
        for name in zf.namelist():
            print(" ", name)
