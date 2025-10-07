import os
import glob

def read_txt_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path):
    try:
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except ImportError:
        return "[Cần cài PyPDF2 để đọc PDF]"

def read_docx(path):
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except ImportError:
        return "[Cần cài python-docx để đọc DOCX]"

def read_file_by_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".md", ".txt"]:
        return read_txt_md(path)
    elif ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    else:
        return "[Không hỗ trợ định dạng này]"

def load_files(folder):
    patterns = ["*.md", "*.txt", "*.pdf", "*.docx"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    data = []
    for file_path in files:
        content = read_file_by_type(file_path)
        data.append({
            "filename": os.path.basename(file_path),
            "ext": os.path.splitext(file_path)[1].lower(),
            "text": content
        })
    return data
