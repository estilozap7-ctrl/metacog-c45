"""
=========================================================
MetaCog-C45 Framework
Markdown to PDF Converter (Generador de PDF)
=========================================================

Lee 'manual_de_usuario.md' y genera un documento PDF profesional
de alta calidad con tipografía Arial, títulos destacados y bloques
de código en formato Courier.
=========================================================
"""

import os
import re
from fpdf import FPDF


def clean_text(text):
    """Limpia el texto eliminando emojis y caracteres especiales no soportados en Latin-1."""
    # Reemplazar caracteres especiales conocidos
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("🍃", "").replace("📊", "").replace("🌱", "")
    text = text.replace("📋", "").replace("🛠️", "").replace("⚙️", "")
    text = text.replace("💡", "").replace("ℹ️", "").replace("🎯", "")
    text = text.replace("📉", "").replace("🔑", "").replace("🏁", "")
    text = text.replace("🚀", "").replace("🌿", "").replace("✅", "")
    text = text.replace("🎨", "").replace("📁", "").replace("🧬", "")
    text = text.replace("🎲", "").replace("✂️", "").replace("📥", "")
    text = text.replace("🧹", "").replace("🔄", "").replace("🔬", "")
    text = text.replace("🔍", "").replace("📖", "").replace("⚙", "")
    text = text.replace("✔", "V")
    
    # Remover cualquier caracter fuera de Latin-1 para evitar caídas de fpdf2
    cleaned = []
    for char in text:
        try:
            char.encode('latin-1')
            cleaned.append(char)
        except UnicodeEncodeError:
            # Reemplazar caracteres raros por espacios o guiones
            cleaned.append(' ')
    return "".join(cleaned)


class PDFManual(FPDF):

    def __init__(self):
        super().__init__()
        # Cargar fuentes del sistema para soporte Unicode básico
        font_path_regular = r"C:\Windows\Fonts\arial.ttf"
        font_path_bold = r"C:\Windows\Fonts\arialbd.ttf"
        font_path_italic = r"C:\Windows\Fonts\ariali.ttf"
        font_path_mono = r"C:\Windows\Fonts\cour.ttf"

        self.add_font("Arial", "", font_path_regular)
        self.add_font("Arial", "B", font_path_bold)
        self.add_font("Arial", "I", font_path_italic)
        self.add_font("Courier", "", font_path_mono)

    def header(self):
        # Encabezado de página
        if self.page_no() > 1:
            self.set_font("Arial", "I", 8)
            self.set_text_color(113, 113, 122) # Gris oscuro
            self.cell(0, 10, "Manual de Usuario - MetaCog-C45 Framework", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
            # Línea sutil
            self.set_draw_color(108, 99, 255)
            self.set_line_width(0.2)
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        # Pie de página
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(113, 113, 122)
        # Línea sutil
        self.set_draw_color(228, 228, 231)
        self.line(10, 280, 200, 280)
        
        # Número de página centrado
        self.cell(0, 10, f"Página {self.page_no()}", border=0, align="C")


def generate_pdf():
    # Rutas de archivos
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    md_path = os.path.join(base_dir, 'manual_de_usuario.md')
    pdf_path_root = os.path.join(base_dir, 'manual_de_usuario.pdf')
    pdf_path_web = os.path.join(base_dir, 'web_verification', 'manual_de_usuario.pdf')

    print(f"Leyendo manual desde: {md_path}")
    if not os.path.exists(md_path):
        print("Error: No se encontró 'manual_de_usuario.md'")
        return

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Inicializar PDF
    pdf = PDFManual()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Página de Portada
    pdf.add_page()
    pdf.ln(40)
    
    # Título principal de la portada
    pdf.set_font("Arial", "B", 26)
    pdf.set_text_color(108, 99, 255) # Color acento (#6c63ff)
    pdf.cell(0, 15, "MANUAL DE USUARIO", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Subtítulo de la portada
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(16, 18, 26) # Color oscuro
    pdf.cell(0, 12, "Implementación de MetaCog-C45 con Nuevos Datasets", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Separador visual
    pdf.set_draw_color(108, 99, 255)
    pdf.set_line_width(1)
    pdf.line(40, 100, 170, 100)
    
    pdf.ln(40)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(113, 113, 122)
    pdf.cell(0, 8, "Framework de Clasificación C4.5 en Python", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Algoritmos de Construcción, Poda y Evaluación de Modelos", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(60)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(16, 18, 26)
    pdf.cell(0, 6, "Autor: Luis Alberto Buelvas Cogollo", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(113, 113, 122)
    pdf.cell(0, 6, "Framework: MetaCog-C45 (v1.0.0)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Fecha: Julio 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # Iniciar Contenido
    pdf.add_page()
    in_code_block = False
    code_lines = []

    for line in lines:
        stripped = line.strip()

        # Limpiar línea
        cleaned_stripped = clean_text(stripped)

        # Manejo de bloques de código (```...)
        if stripped.startswith("```"):
            if in_code_block:
                # Salir de bloque de código y escribirlo
                pdf.set_font("Courier", "", 8)
                pdf.set_text_color(40, 116, 166) # Azul
                pdf.set_fill_color(245, 245, 247) # Gris claro
                
                # Consolidar líneas de código
                code_text = "".join(code_lines)
                pdf.multi_cell(0, 4, code_text, border=1, fill=True)
                pdf.ln(4)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(clean_text(line))
            continue

        # Título principal (# ...)
        if stripped.startswith("# "):
            title = cleaned_stripped
            pdf.ln(5)
            pdf.set_font("Arial", "B", 18)
            pdf.set_text_color(108, 99, 255)
            pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            continue

        # Subtítulo 1 (## ...)
        if stripped.startswith("## "):
            title = cleaned_stripped
            pdf.ln(6)
            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(16, 18, 26)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            # Subrayado sutil
            pdf.set_draw_color(228, 228, 231)
            pdf.set_line_width(0.5)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(4)
            continue

        # Subtítulo 2 (### ...)
        if stripped.startswith("### "):
            title = cleaned_stripped
            pdf.ln(4)
            pdf.set_font("Arial", "B", 10.5)
            pdf.set_text_color(108, 99, 255)
            pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            continue

        # Citas / Info Boxes (> ...)
        if stripped.startswith("> "):
            text = cleaned_stripped
            pdf.set_font("Arial", "I", 9.5)
            pdf.set_text_color(16, 18, 26)
            pdf.set_fill_color(255, 250, 230) # Fondo crema suave
            pdf.set_draw_color(255, 217, 61) # Borde amarillo
            pdf.set_line_width(0.5)
            pdf.multi_cell(0, 5, text, border="L", fill=True)
            pdf.ln(3)
            continue

        # Listas con viñetas
        if stripped.startswith("* ") or stripped.startswith("- "):
            text = cleaned_stripped
            
            # Formatear texto
            pdf.set_font("Arial", "", 9.5)
            pdf.set_text_color(39, 39, 42)
            
            # Viñeta dibujada
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.fill_ellipse(x + 2, y + 2.5, 1.5, 1.5)
            
            # Escribir texto indentado
            pdf.set_x(x + 6)
            pdf.multi_cell(0, 5, text)
            pdf.ln(1)
            continue

        # Línea horizontal
        if stripped == "---":
            pdf.ln(5)
            pdf.set_draw_color(228, 228, 231)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            continue

        # Líneas vacías
        if not stripped:
            pdf.ln(2)
            continue

        # Párrafo común
        text = cleaned_stripped.replace("**", "").replace("`", "")
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(39, 39, 42)
        pdf.multi_cell(0, 5, text)
        pdf.ln(2.5)

    # Guardar en raíz y carpeta web
    print(f"Exportando PDF en la raíz: {pdf_path_root}")
    pdf.output(pdf_path_root)
    
    print(f"Exportando PDF en la carpeta web: {pdf_path_web}")
    pdf.output(pdf_path_web)
    
    print("¡PDF generado con éxito!")


if __name__ == "__main__":
    generate_pdf()
