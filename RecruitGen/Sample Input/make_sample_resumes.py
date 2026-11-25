#!/usr/bin/env python3
"""
Synthetic resume dataset generator for RecruitGen bulk-processing tests.

Everything produced is FICTIONAL (invented names, reserved example.* e-mail
domains, 555-style phone numbers, invented companies/schools).

Outputs (into --out):
  resumes_batch_78_pdfs.zip      78 PDFs (digital, scanned, multipage, multilingual)
  resumes_batch_100_mixed.zip    exactly 100 files: the 78 PDFs + 12 PNG + 10 JPG scans
  edge_cases.zip                 corrupted / unsupported / duplicate / zip-slip / encrypted ...
  ground_truth.csv               expected text + key fields per file (NOT inside the zips)
  ground_truth.json              same + structured fields (skills / education / experience)

Usage:  python3 make_sample_resumes.py --out ./sample_data [--seed 20261026]
Requires: reportlab, Pillow, numpy
"""
import argparse, csv, io, json, os, random, shutil, unicodedata, zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib import pdfencrypt

PAGE_W, PAGE_H = 595.0, 842.0  # A4 in points
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 50.0, 58.0, 54.0

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/freefont"]
FONT_FAMILIES = {
    "sans":  ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    "free":  ("FreeSans.ttf", "FreeSansBold.ttf"),
}
FONT_PATHS = {}


def find_font(fname):
    for d in FONT_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fname)


def register_fonts():
    for fam, (reg, bold) in FONT_FAMILIES.items():
        for style, fname in (("R", reg), ("B", bold)):
            key = f"{fam}-{style}"
            FONT_PATHS[key] = find_font(fname)
            pdfmetrics.registerFont(TTFont(key, FONT_PATHS[key]))


# --------------------------------------------------------------------------
# Content pools
# --------------------------------------------------------------------------
EN_FIRST = ["Aarav", "Priya", "Liam", "Olivia", "Noah", "Emma", "Rohan", "Ananya", "Ethan", "Sophia",
            "Kabir", "Isha", "Mason", "Ava", "Arjun", "Meera", "Lucas", "Mia", "Vikram", "Nisha",
            "Daniel", "Chloe", "Sameer", "Tara", "Henry", "Grace", "Rahul", "Kavya", "Oliver", "Zoe"]
EN_LAST = ["Sharma", "Patel", "Nguyen", "Carter", "Bennett", "Rao", "Kulkarni", "Hughes", "Foster", "Mehta",
           "Reynolds", "Iyer", "Brooks", "Desai", "Walker", "Joshi", "Hayes", "Kapoor", "Morgan", "Banerjee",
           "Fischer", "Chandra", "Ellison", "Pillai", "Sutton", "Naidu", "Whitaker", "Menon", "Prescott", "Gill"]
LANG_NAMES = {
    "es": (["José", "María", "Álvaro", "Lucía", "Sofía", "Héctor", "Inés", "Andrés"],
           ["Álvarez", "Fernández", "Muñoz", "Gutiérrez", "Peña", "Domínguez", "Núñez", "Ibáñez"]),
    "fr": (["Hélène", "François", "Amélie", "Étienne", "Clémence", "Gaëtan", "Léa", "Rémi"],
           ["Dubois", "Lefèvre", "Moreau", "Garçon", "Renaud", "Bélanger", "Thévenet", "Côté"]),
    "de": (["Jürgen", "Käthe", "Björn", "Märta", "Lukas", "Anneliese", "Jörg", "Frieda"],
           ["Müller", "Schröder", "Köhler", "Weiß", "Böhm", "Grünewald", "Förster", "Hübner"]),
    "pt": (["João", "Conceição", "Tiago", "Beatriz", "Gonçalo", "Inês", "Rúben", "Luíza"],
           ["Gonçalves", "Magalhães", "Araújo", "Cardoso", "Fonseca", "Pimentel", "Cavalcanti", "Sá"]),
}
COMPANIES = ["Brightwave Systems", "Nimbus Analytics", "Harborlight Labs", "Verdant Logistics", "Quillon Software",
             "Atlas & Finch", "Corvid Digital", "Meridian Health Tech", "Pinecrest Financial", "Lumen Retail Group",
             "Orbitron Robotics", "Saffron Foods Ltd", "Bluepeak Consulting", "Granite Works", "Tidewater Media",
             "Redbird Payments", "Kestrel Energy", "Cobalt Cloud", "Marlowe & Sons", "Evergreen Education"]
SCHOOLS = {
    "en": ["Northfield Institute of Technology", "Lakeshore University", "Westbridge College",
           "Riverton State University", "Summit Polytechnic"],
    "es": ["Universidad Politécnica de Valmora", "Universidad del Litoral Andino"],
    "fr": ["Université de Montclair-sur-Loire", "École Supérieure de Rochebrune"],
    "de": ["Technische Universität Brandenburg-Nord", "Hochschule Elbtal"],
    "pt": ["Universidade Federal do Vale Verde", "Instituto Politécnico de Serra Azul"],
}
CITIES = {
    "en": ["Pune, India", "Austin, TX", "Leeds, UK", "Toronto, Canada", "Bengaluru, India", "Denver, CO", "Sydney, Australia"],
    "es": ["Valmora, España", "Quito, Ecuador", "Córdoba, Argentina"],
    "fr": ["Lyon, France", "Montréal, Canada", "Liège, Belgique"],
    "de": ["Köln, Deutschland", "Zürich, Schweiz", "Graz, Österreich"],
    "pt": ["Lisboa, Portugal", "São Paulo, Brasil", "Porto, Portugal"],
}

# title -> (en area phrase, skill pool, degrees)
EN_TITLES = {
    "Software Engineer": ("backend and web application development",
                          ["Python", "Java", "SQL", "REST APIs", "Docker", "Git", "Kubernetes", "PostgreSQL", "Redis", "CI/CD", "Linux", "TypeScript"]),
    "Data Scientist": ("machine learning and statistical modelling",
                       ["Python", "scikit-learn", "TensorFlow", "PyTorch", "SQL", "Pandas", "NLP", "Feature engineering", "A/B testing", "Spark", "MLflow", "Statistics"]),
    "Data Analyst": ("business intelligence and reporting",
                     ["SQL", "Excel", "Power BI", "Tableau", "Python", "Data cleaning", "Dashboards", "Statistics", "ETL", "Google Analytics", "R", "Storytelling"]),
    "DevOps Engineer": ("cloud infrastructure and release automation",
                        ["AWS", "Terraform", "Docker", "Kubernetes", "Jenkins", "Ansible", "Prometheus", "Grafana", "Bash", "Linux", "GitHub Actions", "Python"]),
    "Product Manager": ("roadmap planning and stakeholder management",
                        ["Roadmapping", "User research", "Agile", "Jira", "SQL", "A/B testing", "Prioritisation", "OKRs", "Wireframing", "Analytics", "Stakeholder management", "Go-to-market"]),
    "UX Designer": ("user-centred design and prototyping",
                    ["Figma", "User research", "Wireframing", "Prototyping", "Design systems", "Usability testing", "Accessibility", "HTML/CSS", "Information architecture", "Sketch", "Journey mapping", "Workshops"]),
    "HR Manager": ("talent acquisition and employee engagement",
                   ["Recruitment", "Onboarding", "Payroll", "Employee relations", "HRIS", "Performance reviews", "Labour law", "Training design", "Interviewing", "Excel", "Conflict resolution", "Policy writing"]),
    "Accountant": ("financial reporting and compliance",
                   ["Tally", "QuickBooks", "Excel", "GST", "Auditing", "Reconciliation", "Budgeting", "IFRS", "Accounts payable", "Taxation", "SAP FICO", "Forecasting"]),
    "Marketing Manager": ("digital campaigns and brand growth",
                          ["SEO", "Google Ads", "Content strategy", "Email marketing", "HubSpot", "Social media", "Copywriting", "Analytics", "Brand positioning", "Canva", "CRM", "Event planning"]),
    "Sales Executive": ("B2B lead generation and account growth",
                        ["Lead generation", "Salesforce", "Negotiation", "Cold outreach", "CRM", "Forecasting", "Presentations", "Account management", "Upselling", "Pipeline management", "Excel", "Customer success"]),
    "Mechanical Engineer": ("product design and manufacturing support",
                            ["AutoCAD", "SolidWorks", "ANSYS", "GD&T", "Lean manufacturing", "CNC", "Six Sigma", "Thermodynamics", "Prototyping", "Root cause analysis", "MATLAB", "Quality control"]),
    "Business Analyst": ("requirements gathering and process improvement",
                         ["Requirements gathering", "SQL", "Process mapping", "JIRA", "UML", "Excel", "Power BI", "User stories", "Stakeholder interviews", "Agile", "Visio", "Gap analysis"]),
}
DEGREES_EN = ["B.Tech in Computer Science", "B.Sc. in Statistics", "MBA in Marketing", "B.Com in Accounting",
              "M.Sc. in Data Science", "B.E. in Mechanical Engineering", "BA in Psychology", "M.Tech in Software Engineering",
              "B.Des in Interaction Design", "MBA in Human Resources"]
TITLES_L = {
    "es": ["Ingeniero de Software", "Analista de Datos", "Gerente de Proyectos", "Diseñador UX"],
    "fr": ["Ingénieur logiciel", "Analyste de données", "Chef de projet", "Designer UX"],
    "de": ["Softwareentwickler", "Datenanalyst", "Projektleiter", "UX-Designer"],
    "pt": ["Engenheiro de Software", "Analista de Dados", "Gerente de Projetos", "Designer de UX"],
}
SKILLS_L = ["Python", "SQL", "Excel", "Power BI", "Git", "Docker", "Scrum", "Jira", "Figma", "Tableau", "Linux", "API REST"]
DEGREES_L = {
    "es": ["Grado en Ingeniería Informática", "Máster en Análisis de Datos", "Grado en Administración de Empresas"],
    "fr": ["Licence en Informatique", "Master en Science des données", "Diplôme d'ingénieur en génie logiciel"],
    "de": ["Bachelor Informatik", "Master Wirtschaftsinformatik", "Diplom-Ingenieur Softwaretechnik"],
    "pt": ["Licenciatura em Engenharia Informática", "Mestrado em Ciência de Dados", "Licenciatura em Gestão"],
}
HEADINGS = {
    "en": dict(summary="SUMMARY", skills="SKILLS", exp="EXPERIENCE", edu="EDUCATION", cert="CERTIFICATIONS", proj="PROJECTS"),
    "es": dict(summary="RESUMEN PROFESIONAL", skills="COMPETENCIAS", exp="EXPERIENCIA LABORAL", edu="FORMACIÓN ACADÉMICA", cert="CERTIFICACIONES", proj="PROYECTOS"),
    "fr": dict(summary="PROFIL", skills="COMPÉTENCES", exp="EXPÉRIENCE PROFESSIONNELLE", edu="FORMATION", cert="CERTIFICATIONS", proj="PROJETS"),
    "de": dict(summary="PROFIL", skills="KENNTNISSE", exp="BERUFSERFAHRUNG", edu="AUSBILDUNG", cert="ZERTIFIZIERUNGEN", proj="PROJEKTE"),
    "pt": dict(summary="RESUMO PROFISSIONAL", skills="COMPETÊNCIAS", exp="EXPERIÊNCIA PROFISSIONAL", edu="FORMAÇÃO ACADÉMICA", cert="CERTIFICAÇÕES", proj="PROJETOS"),
}
PRESENT = dict(en="Present", es="Actualidad", fr="Aujourd'hui", de="Heute", pt="Atualidade")
SKILLS_LABEL = dict(en="Core skills", es="Competencias clave", fr="Compétences clés", de="Kernkompetenzen", pt="Competências principais")
YEARS_WORD = dict(en="years", es="años", fr="ans", de="Jahre", pt="anos")

SUMMARY_T = {
    "en": ["{title} with {y} years of experience in {area}. Known for clear communication, reliable delivery and a practical approach to solving problems. Skilled in {a}, {b} and {c}.",
           "Results-driven {title} with {y}+ years in {area}. Comfortable working with distributed teams and turning ambiguous requirements into working solutions using {a} and {b}.",
           "{title} focused on {area}. Over {y} years of hands-on work with {a}, {b} and {c}, supported by a record of mentoring colleagues and meeting deadlines."],
    "es": ["{title} con {y} años de experiencia en entornos colaborativos. Comunicación clara, entrega fiable y enfoque práctico. Dominio de {a}, {b} y {c}.",
           "Profesional orientado a resultados con más de {y} años como {title}. Experiencia en {a} y {b}, y en la coordinación de equipos distribuidos."],
    "fr": ["{title} avec {y} ans d'expérience dans des environnements collaboratifs. Communication claire, livraison fiable et approche pragmatique. Maîtrise de {a}, {b} et {c}.",
           "Professionnel orienté résultats, plus de {y} ans comme {title}. Expérience de {a} et {b}, ainsi que de la coordination d'équipes à distance."],
    "de": ["{title} mit {y} Jahren Erfahrung in kollaborativen Umgebungen. Klare Kommunikation, zuverlässige Lieferung und pragmatischer Ansatz. Sicher in {a}, {b} und {c}.",
           "Ergebnisorientierter {title} mit über {y} Jahren Berufserfahrung. Praxis mit {a} und {b} sowie in der Koordination verteilter Teams."],
    "pt": ["{title} com {y} anos de experiência em ambientes colaborativos. Comunicação clara, entregas fiáveis e abordagem prática. Domínio de {a}, {b} e {c}.",
           "Profissional orientado para resultados com mais de {y} anos como {title}. Experiência em {a} e {b} e na coordenação de equipas remotas."],
}
BULLET_T = {
    "en": ["Designed and delivered {a} solutions that reduced processing time by {n}%.",
           "Collaborated with cross-functional teams to ship {m} releases using {a} and {b}.",
           "Mentored {k} junior team members and led weekly review sessions.",
           "Automated reporting workflows with {a}, saving about {k} hours per week.",
           "Improved customer satisfaction scores by {n}% through data-driven process changes.",
           "Managed a budget of ${m}K and delivered the project {k} weeks ahead of schedule.",
           "Built dashboards in {a} that gave leadership real-time visibility into key metrics.",
           "Migrated legacy systems to {a}, cutting operating costs by {n}%.",
           "Wrote documentation and training material adopted by {m} regional offices.",
           "Identified and resolved {m} critical issues during quarterly audits."],
    "es": ["Diseñé e implementé soluciones con {a} que redujeron el tiempo de procesamiento un {n}%.",
           "Colaboré con equipos multidisciplinares en {m} lanzamientos usando {a} y {b}.",
           "Mentoricé a {k} compañeros junior y dirigí sesiones semanales de revisión.",
           "Automaticé informes con {a}, ahorrando unas {k} horas por semana.",
           "Mejoré la satisfacción de los clientes un {n}% mediante cambios basados en datos."],
    "fr": ["Conçu et déployé des solutions {a} qui ont réduit le temps de traitement de {n} %.",
           "Collaboré avec des équipes transverses sur {m} livraisons avec {a} et {b}.",
           "Accompagné {k} collègues juniors et animé des revues hebdomadaires.",
           "Automatisé les rapports avec {a}, soit environ {k} heures économisées par semaine.",
           "Amélioré la satisfaction client de {n} % grâce à des décisions fondées sur les données."],
    "de": ["Lösungen mit {a} entworfen und umgesetzt, die die Bearbeitungszeit um {n} % senkten.",
           "Mit bereichsübergreifenden Teams {m} Releases mit {a} und {b} ausgeliefert.",
           "{k} Nachwuchskräfte betreut und wöchentliche Review-Runden geleitet.",
           "Berichte mit {a} automatisiert und rund {k} Stunden pro Woche eingespart.",
           "Die Kundenzufriedenheit durch datenbasierte Maßnahmen um {n} % gesteigert."],
    "pt": ["Conceção e entrega de soluções com {a} que reduziram o tempo de processamento em {n}%.",
           "Colaboração com equipas multidisciplinares em {m} lançamentos com {a} e {b}.",
           "Mentoria de {k} colegas juniores e condução de revisões semanais.",
           "Automatização de relatórios com {a}, poupando cerca de {k} horas por semana.",
           "Aumento da satisfação dos clientes em {n}% através de decisões baseadas em dados."],
}
PROJECT_T = {
    "en": ["{a} based internal tool used by {m} teams", "Open-source contribution: {b} plugin for {a}",
           "Capstone: {a} pipeline processing {k}K records daily"],
    "es": ["Herramienta interna con {a} utilizada por {m} equipos", "Proyecto final: flujo de datos con {a} y {b}"],
    "fr": ["Outil interne basé sur {a} utilisé par {m} équipes", "Projet de fin d'études : pipeline {a} et {b}"],
    "de": ["Internes Werkzeug auf Basis von {a}, genutzt von {m} Teams", "Abschlussprojekt: Datenpipeline mit {a} und {b}"],
    "pt": ["Ferramenta interna baseada em {a} usada por {m} equipas", "Projeto final: pipeline de dados com {a} e {b}"],
}
CERTS = ["AWS Certified Cloud Practitioner", "Google Data Analytics Certificate", "PMP", "Scrum Master (PSM I)",
         "Microsoft Azure Fundamentals", "Six Sigma Green Belt", "SHRM-CP", "Tableau Desktop Specialist",
         "Certified Ethical Practices Associate", "ITIL 4 Foundation"]


def ascii_fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace(" ", "")


def make_phone(rng):
    f = rng.choice([0, 1, 2, 3])
    if f == 0:
        return f"+1 (555) 01{rng.randint(0,9)}-{rng.randint(1000,9999)}"
    if f == 1:
        return f"+44 20 7946 0{rng.randint(100,999)}"
    if f == 2:
        return f"+91 90000 0{rng.randint(1000,9999)}"
    return f"555-01{rng.randint(10,99)}"


def month_year(rng, y):
    return f"{rng.randint(1,12):02d}/{y}"


def build_resume(rng, lang, used_names, long_form=False, short_form=False):
    """Return dict with structured content + the ordered list of (kind, text) lines."""
    while True:
        if lang == "en":
            first, last = rng.choice(EN_FIRST), rng.choice(EN_LAST)
        else:
            fs, ls = LANG_NAMES[lang]
            first, last = rng.choice(fs), rng.choice(ls)
        if (first, last) not in used_names:
            used_names.add((first, last))
            break
    name = f"{first} {last}"
    email = f"{ascii_fold(first)}.{ascii_fold(last)}@{rng.choice(['example.com','example.org','example.net'])}"
    phone = make_phone(rng)
    location = rng.choice(CITIES[lang])

    if lang == "en":
        title = rng.choice(list(EN_TITLES))
        area, pool = EN_TITLES[title]
        degree_pool = DEGREES_EN
    else:
        title = rng.choice(TITLES_L[lang])
        area, pool = "", SKILLS_L
        degree_pool = DEGREES_L[lang]

    years = rng.randint(2, 14)
    skills = rng.sample(pool, 8)
    a, b, c = skills[0], skills[1], skills[2]
    summary = rng.choice(SUMMARY_T[lang]).format(title=title, y=years, area=area, a=a, b=b, c=c)

    n_jobs = 1 if short_form else (rng.randint(7, 8) if long_form else rng.randint(2, 3))
    n_bul = 2 if short_form else (rng.randint(5, 6) if long_form else rng.randint(3, 4))
    exp = []
    end_year = 2026
    for _ in range(n_jobs):
        dur = rng.randint(1, 4)
        start = end_year - dur
        end = PRESENT[lang] if not exp else month_year(rng, end_year)
        tpl = rng.sample(BULLET_T[lang], min(n_bul, len(BULLET_T[lang])))
        bullets = [t.format(a=rng.choice(skills), b=rng.choice(skills), n=rng.randint(8, 65),
                            m=rng.randint(3, 40), k=rng.randint(2, 20)) for t in tpl]
        role = title if not exp else rng.choice([f"Junior {title}", f"Associate {title}", title] if lang == "en" else [title])
        exp.append(dict(role=role, company=rng.choice(COMPANIES),
                        dates=f"{month_year(rng, start)} - {end}", bullets=bullets))
        end_year = start
    edu = [dict(degree=rng.choice(degree_pool), school=rng.choice(SCHOOLS[lang]), year=str(2026 - years - rng.randint(1, 4)))]
    certs = [] if short_form else rng.sample(CERTS, rng.randint(1, 3))
    projects = []
    if long_form:
        projects = [t.format(a=rng.choice(skills), b=rng.choice(skills), m=rng.randint(2, 12), k=rng.randint(5, 90))
                    for t in rng.sample(PROJECT_T[lang], min(2, len(PROJECT_T[lang])))]

    H = HEADINGS[lang]
    lines = [("name", name), ("title", title), ("contact", f"{email} | {phone} | {location}"),
             ("heading", H["summary"]), ("text", summary),
             ("heading", H["skills"]), ("text", f"{SKILLS_LABEL[lang]}: " + ", ".join(skills)),
             ("heading", H["exp"])]
    for j in exp:
        lines.append(("subhead", f"{j['role']} - {j['company']} ({j['dates']})"))
        lines += [("bullet", f"- {bl}") for bl in j["bullets"]]
    lines.append(("heading", H["edu"]))
    for e in edu:
        lines.append(("text", f"{e['degree']}, {e['school']}, {e['year']}"))
    if projects:
        lines.append(("heading", H["proj"]))
        lines += [("bullet", f"- {p}") for p in projects]
    if certs:
        lines.append(("heading", H["cert"]))
        lines += [("bullet", f"- {c_}") for c_ in certs]

    return dict(name=name, email=email, phone=phone, location=location, title=title, language=lang,
                skills=skills, education=edu, experience=exp, certifications=certs, lines=lines)


# --------------------------------------------------------------------------
# Layout (shared by PDF-text and raster renderers so ground truth is exact)
# --------------------------------------------------------------------------
STYLE = dict(
    name=("B", 20, 1.25), title=("R", 12, 1.4), contact=("R", 9.5, 1.5), heading=("B", 11, 1.5),
    subhead=("B", 10, 1.45), text=("R", 10, 1.4), bullet=("R", 10, 1.4),
)
ACCENTS = [(0.10, 0.20, 0.45), (0.45, 0.10, 0.15), (0.05, 0.35, 0.25), (0.25, 0.25, 0.25), (0.35, 0.15, 0.50)]


def wrap(text, font, size, max_w, indent_w=0.0):
    words, out, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= max_w - (indent_w if out else 0):
            cur = trial
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def layout(resume, fam, centered, accent):
    """Return (pages, gt_lines). pages: list of list of draw ops in PDF points (origin bottom-left)."""
    pages, ops = [], []
    y = PAGE_H - MARGIN_TOP
    gt_lines = []
    max_w = PAGE_W - 2 * MARGIN_X

    def new_page():
        nonlocal ops, y
        pages.append(ops)
        ops, y = [], PAGE_H - MARGIN_TOP

    for kind, text in resume["lines"]:
        style, size, lead = STYLE[kind]
        font = f"{fam}-{style}"
        indent = 12 if kind == "bullet" else 0
        pre = (8 if kind == "heading" else 3 if kind == "subhead" else 0)
        wrapped = wrap(text, font, size, max_w - indent, indent_w=8 if kind == "bullet" else 0)
        need = pre + len(wrapped) * size * lead + (4 if kind == "heading" else 0)
        if y - need < MARGIN_BOTTOM:
            new_page()
        y -= pre
        for i, ln in enumerate(wrapped):
            y -= size * lead
            x = MARGIN_X + indent + (8 if (kind == "bullet" and i > 0) else 0)
            if centered and kind in ("name", "title", "contact"):
                x = (PAGE_W - pdfmetrics.stringWidth(ln, font, size)) / 2
            color = accent if kind in ("name", "heading") else (0.1, 0.1, 0.1)
            ops.append(dict(op="text", x=x, y=y, text=ln, font=font, size=size, color=color))
            gt_lines.append(ln)
        if kind == "heading":
            ops.append(dict(op="rule", x1=MARGIN_X, x2=PAGE_W - MARGIN_X, y=y - 3, color=accent))
            y -= 4
    pages.append(ops)
    return pages, gt_lines


def render_pdf_digital(pages, resume, path, encrypt_pw=None):
    enc = pdfencrypt.StandardEncryption(encrypt_pw, canPrint=1) if encrypt_pw else None
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), encrypt=enc)
    c.setTitle(f"Resume - {resume['name']}")
    c.setAuthor(resume["name"])
    for ops in pages:
        for o in ops:
            if o["op"] == "text":
                c.setFillColorRGB(*o["color"])
                c.setFont(o["font"], o["size"])
                c.drawString(o["x"], o["y"], o["text"])
            else:
                c.setStrokeColorRGB(*o["color"])
                c.setLineWidth(0.8)
                c.line(o["x1"], o["y"], o["x2"], o["y"])
        c.showPage()
    c.save()


def rasterize_page(ops, dpi=150):
    s = dpi / 72.0
    img = Image.new("L", (int(PAGE_W * s), int(PAGE_H * s)), 255)
    d = ImageDraw.Draw(img)
    fonts = {}
    for o in ops:
        if o["op"] == "text":
            key = (o["font"], o["size"])
            if key not in fonts:
                fonts[key] = ImageFont.truetype(FONT_PATHS[o["font"]], int(round(o["size"] * s)))
            gray = int(0.299 * o["color"][0] * 255 + 0.587 * o["color"][1] * 255 + 0.114 * o["color"][2] * 255)
            d.text((o["x"] * s, (PAGE_H - o["y"]) * s), o["text"], font=fonts[key], fill=gray, anchor="ls")
        else:
            gray = int(0.3 * 255 * 0.5)
            d.line([(o["x1"] * s, (PAGE_H - o["y"]) * s), (o["x2"] * s, (PAGE_H - o["y"]) * s)], fill=gray, width=max(1, int(s * 0.8)))
    return img


SEVERITY = {
    "mild":   dict(rot=0.6, blur=0.5, noise=6,  contrast=0.92, shadow=0.00, down=1.0, jpeg=85),
    "medium": dict(rot=1.2, blur=0.7, noise=8,  contrast=0.85, shadow=0.10, down=0.85, jpeg=65),
    "heavy":  dict(rot=2.0, blur=1.0, noise=14, contrast=0.72, shadow=0.18, down=0.70, jpeg=45),
}


def degrade(img, level, rng):
    p = SEVERITY[level]
    w, h = img.size
    ang = rng.uniform(-p["rot"], p["rot"])
    img = img.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if p["down"] < 1.0:
        img = img.resize((int(w * p["down"]), int(h * p["down"])), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    arr = np.asarray(img, dtype=np.float32)
    arr = 255 - (255 - arr) * p["contrast"]                       # wash out ink
    if p["shadow"] > 0:                                            # uneven lighting
        gx = np.linspace(0, 1, w)[None, :]
        gy = np.linspace(0, 1, h)[:, None]
        arr = arr * (1 - p["shadow"] * (0.6 * gx + 0.4 * gy))
    nrng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = arr + nrng.normal(0, p["noise"], arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")


def jpeg_bytes(img, q):
    bio = io.BytesIO()
    img.convert("RGB").save(bio, "JPEG", quality=q)
    return bio.getvalue()


def render_pdf_scanned(pages, resume, path, level, rng):
    p = SEVERITY[level]
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Scanned document")           # no name/author metadata on scans
    for ops in pages:
        img = degrade(rasterize_page(ops), level, rng)
        c.drawImage(ImageReader(io.BytesIO(jpeg_bytes(img, p["jpeg"]))), 0, 0, PAGE_W, PAGE_H)
        c.showPage()
    c.save()


def render_image(pages, path, fmt, level, rng):
    p = SEVERITY[level]
    img = degrade(rasterize_page(pages[0]), level, rng)
    if fmt == "PNG":
        img.save(path, "PNG", optimize=True)          # grayscale keeps files small
    else:
        img.save(path, "JPEG", quality=p["jpeg"])


def norm(s):
    return " ".join(s.split())


# --------------------------------------------------------------------------
def file_name(rng, resume, idx, used):
    first, last = resume["name"].split(" ", 1)
    styles = [f"{first}_{last}_Resume", f"CV_{first} {last}", f"{last.lower()}_{first.lower()}_cv_2026",
              f"{first}{last}-Resume-{rng.randint(2023, 2026)}", f"Resume - {first} {last}"]
    base = rng.choice(styles).replace(" ", " ")
    base = unicodedata.normalize("NFC", base)
    nm = base
    while nm.lower() in used:
        nm = f"{base}_{idx}"
    used.add(nm.lower())
    return nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./sample_data")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--multilingual", action="store_true",
                    help="include es/fr/de/pt resumes (default: all English)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    register_fonts()
    out = Path(args.out)
    work = out / "_work"
    if work.exists():
        shutil.rmtree(work)
    (work / "pdf").mkdir(parents=True)
    (work / "img").mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)

    used_names, used_files, truth = set(), set(), []

    # ---- plan of the 78 PDFs ------------------------------------------------
    plan = []
    langs = (["es"] * 4 + ["fr"] * 4 + ["de"] * 4 + ["pt"] * 4 + ["en"] * 62) if args.multilingual else ["en"] * 78
    scanned_levels = ["mild"] * 5 + ["medium"] * 5 + ["heavy"] * 4
    rng.shuffle(langs)
    idxs = list(range(78))
    rng.shuffle(idxs)
    scan_idx = {i: scanned_levels[k] for k, i in enumerate(idxs[:14])}
    long_idx = set(idxs[14:30])  # 16 multipage digital
    scan_multi = set(idxs[:2])   # 2 multipage scans
    for i in range(78):
        plan.append(dict(lang=langs[i], scan=scan_idx.get(i), long=(i in long_idx or i in scan_multi)))

    def emit(i, spec, ext, short=False):
        r = build_resume(rng, spec["lang"], used_names, long_form=spec.get("long", False), short_form=short)
        fam = rng.choice(list(FONT_FAMILIES))
        pages, gt_lines = layout(r, fam, centered=rng.random() < 0.4, accent=rng.choice(ACCENTS))
        base = file_name(rng, r, i, used_files)
        fname = f"{base}.{ext}"
        level = spec.get("scan")
        if ext == "pdf":
            target = work / "pdf" / fname
            if level:
                render_pdf_scanned(pages, r, target, level, rng)
                kind = f"scanned_{level}"
            else:
                render_pdf_digital(pages, r, target)
                kind = "digital"
        else:
            target = work / "img" / fname
            render_image(pages, target, "PNG" if ext == "png" else "JPEG", level, rng)
            kind = f"scanned_{level}"
        truth.append(dict(
            filename=fname, kind=kind, language=r["language"], pages=len(pages) if ext == "pdf" else 1,
            name=r["name"], email=r["email"], phone=r["phone"],
            expected_text="\n".join(gt_lines),
            structured=dict(title=r["title"], skills=r["skills"], education=r["education"],
                            experience=r["experience"], certifications=r["certifications"]),
        ))
        return fname, pages

    # PDFs (78)
    for i, spec in enumerate(plan):
        emit(i, spec, "pdf")

    # 22 extra single-page image resumes for the 100-file mixed batch
    levels_img = ["mild"] * 8 + ["medium"] * 8 + ["heavy"] * 6
    for k in range(22):
        ext = "png" if k < 12 else "jpg"
        emit(100 + k, dict(lang="en", scan=levels_img[k], long=False), ext, short=True)

    # NOTE: image resumes use short_form content, which always fits on a single page,
    # so their expected_text is the complete text of the image.
    for t in truth:
        if t["filename"].lower().endswith((".png", ".jpg")):
            assert t["pages"] == 1

    # ---- ZIPs -----------------------------------------------------------------------------------
    pdfs = sorted((work / "pdf").iterdir())
    imgs = sorted((work / "img").iterdir())
    assert len(pdfs) == 78 and len(imgs) == 22, (len(pdfs), len(imgs))
    with zipfile.ZipFile(out / "resumes_batch_78_pdfs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in pdfs:
            z.write(p, p.name)
    with zipfile.ZipFile(out / "resumes_batch_100_mixed.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in pdfs + imgs:
            z.write(p, p.name)

    # ---- edge cases ------------------------------------------------------------------------------
    good = [pdfs[0], next(p for p in pdfs if "scanned" in next(t["kind"] for t in truth if t["filename"] == p.name)),
            next(p for p in pdfs if next(t["pages"] for t in truth if t["filename"] == p.name) == 2)]  # a multipage one
    blank = work / "blank_page.pdf"
    c = canvas.Canvas(str(blank), pagesize=(PAGE_W, PAGE_H)); c.showPage(); c.save()
    enc = work / "encrypted.pdf"
    rr = build_resume(rng, "en", used_names)
    pg, _ = layout(rr, "sans", False, ACCENTS[0]); render_pdf_digital(pg, rr, enc, encrypt_pw="s3cret")
    real = pdfs[3].read_bytes()
    with zipfile.ZipFile(out / "edge_cases.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for g in good:
            z.write(g, g.name)                                                   # 3 valid files
        z.writestr("corrupted_truncated.pdf", real[: len(real) // 2])             # cut in half
        z.writestr("corrupted_garbage.pdf", os.urandom(4096))                     # random bytes
        z.writestr("fake_image.png", b"this is plain text pretending to be a png") # bad magic
        z.writestr("empty.pdf", b"")                                              # zero bytes
        z.writestr("notes.txt", "not a resume format we accept")                  # unsupported ext
        z.writestr("resume.docx", b"PK\x03\x04 fake docx")                        # unsupported ext
        z.writestr("installer.exe", b"MZ\x90\x00 fake exe")                       # dangerous ext
        z.write(blank, "blank_page.pdf")                                          # valid pdf, no text
        z.write(enc, "encrypted.pdf")                                             # password protected
        z.writestr("teamA/John_Doe_Resume.pdf", real)                             # duplicate basename
        z.writestr("teamB/John_Doe_Resume.pdf", pdfs[5].read_bytes())             # duplicate basename
        z.writestr("../../evil_zip_slip.pdf", real)                               # path traversal
        z.writestr("/abs/evil_absolute.pdf", real)                                # absolute path
        z.writestr("__MACOSX/._junk.pdf", b"\x00\x05\x16\x07")                    # mac metadata junk
    # ---- ground truth ----------------------------------------------------------------------------
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "kind", "language", "pages", "name", "email", "phone", "expected_text"])
        for t in truth:
            w.writerow([t["filename"], t["kind"], t["language"], t["pages"], t["name"], t["email"], t["phone"], t["expected_text"]])
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    shutil.rmtree(work)
    print("pdfs:", len(pdfs), "images:", len(imgs), "truth rows:", len(truth))


if __name__ == "__main__":
    main()
