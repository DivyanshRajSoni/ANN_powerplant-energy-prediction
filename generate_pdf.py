import re
from fpdf import FPDF

class DocPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Power Plant Energy Output Prediction - Project Documentation", align="C")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title, level=1):
        sizes = {1: 18, 2: 15, 3: 13}
        size = sizes.get(level, 12)
        self.ln(4)
        self.set_font("Helvetica", "B", size)
        self.set_text_color(30, 60, 120)
        self.multi_cell(0, size * 0.6, title)
        if level == 1:
            self.set_draw_color(30, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(50, 50, 50)
        self.set_draw_color(200, 200, 200)
        x = self.get_x()
        y = self.get_y()
        lines = code.strip().split("\n")
        line_h = 5
        block_h = len(lines) * line_h + 6
        if y + block_h > 270:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, 190, block_h, style="DF")
        self.ln(3)
        for line in lines:
            safe = line.encode('latin-1', 'replace').decode('latin-1')
            self.cell(5)
            self.cell(0, line_h, safe)
            self.ln(line_h)
        self.ln(3)

    def table(self, headers, rows):
        col_count = len(headers)
        col_w = 190 / col_count
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_w, 7, h.strip(), border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            self.set_fill_color(240, 245, 255) if fill else self.set_fill_color(255, 255, 255)
            max_lines = 1
            for cell_text in row:
                lines_needed = max(1, len(cell_text.strip()) // int(col_w / 2) + 1)
                max_lines = max(max_lines, lines_needed)
            row_h = 7 * max_lines
            for cell_text in row:
                safe = cell_text.strip().encode('latin-1', 'replace').decode('latin-1')
                self.cell(col_w, row_h, safe, border=1, fill=fill, align="L")
            self.ln()
            fill = not fill

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(8)
        bullet_char = chr(149)
        safe = text.encode('latin-1', 'replace').decode('latin-1')
        self.cell(0, 6, f"{bullet_char}  {safe}")
        self.ln(7)


def parse_table(lines):
    headers = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip separator
        row = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(row)
    return headers, rows


def build_pdf(md_path, out_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(0, 14, "Project Documentation", align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 10, "Power Plant Energy Output Prediction\nUsing Artificial Neural Networks (PyTorch)", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "ANN Regression | Deep Learning | Combined Cycle Power Plant", align="C")
    pdf.ln(40)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())

    pdf.add_page()

    lines = content.split("\n")
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]

        # Code block toggle
        if line.strip().startswith("```"):
            if in_code:
                pdf.code_block("\n".join(code_buf))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1]):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            headers, rows = parse_table(table_lines)
            pdf.table(headers, rows)
            pdf.ln(3)
            continue

        # Headings
        if line.startswith("# ") and not line.startswith("## "):
            # Skip the main title, we have a title page
            i += 1
            continue
        if line.startswith("## "):
            pdf.section_title(line[3:].strip(), level=1)
            i += 1
            continue
        if line.startswith("### "):
            pdf.section_title(line[4:].strip(), level=2)
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            i += 1
            continue

        # Bullet points
        if line.strip().startswith("- "):
            text = line.strip()[2:]
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # strip bold markdown
            text = re.sub(r"`(.*?)`", r"\1", text)  # strip inline code
            pdf.bullet(text)
            i += 1
            continue

        # Numbered list
        if re.match(r"^\d+\.\s", line.strip()):
            text = re.sub(r"^\d+\.\s+", "", line.strip())
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
            text = re.sub(r"`(.*?)`", r"\1", text)
            pdf.bullet(text)
            i += 1
            continue

        # Bold line
        bold_match = re.match(r"^\*\*(.*?)\*\*\s*(.*)", line.strip())
        if bold_match and not line.strip().startswith("-"):
            text = bold_match.group(1) + " " + bold_match.group(2)
            text = re.sub(r"`(.*?)`", r"\1", text)
            pdf.bold_text(text.strip())
            i += 1
            continue

        # Normal text
        text = line.strip()
        if text:
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
            text = re.sub(r"`(.*?)`", r"\1", text)
            text = re.sub(r"\$(.*?)\$", r"\1", text)
            pdf.body_text(text)

        i += 1

    pdf.output(out_path)
    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    build_pdf(
        r"c:\Users\lenovo\Prime_classes\DeepLearning\powerplant\PROJECT_DOCUMENTATION.md",
        r"c:\Users\lenovo\Prime_classes\DeepLearning\powerplant\PROJECT_DOCUMENTATION.pdf"
    )
