#!/usr/bin/env python3
"""Genera docs/Makima_Guida_Amichevole.pdf — guida senza tecnicismi."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

OUT = Path(__file__).resolve().parent / "Makima_Guida_Amichevole.pdf"

# Tipografia
pdfmetrics.registerFont(TTFont("NotoSerif", "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSerif-Bold", "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NotoSerif-Italic", "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NotoSans-Italic", "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf"))

# Palette: carta fredda + teal bosco (niente viola, niente cream terracotta)
INK = HexColor("#1A2428")
MUTED = HexColor("#4A5A60")
TEAL = HexColor("#0E6B6E")
TEAL_DEEP = HexColor("#0A4548")
AMBER = HexColor("#C47A2C")
PAPER = HexColor("#F3F6F7")
BAND = HexColor("#E4EEED")
RULE = HexColor("#B8CBCA")


class Rule(Flowable):
    def __init__(self, width=None, stroke=RULE, thickness=0.6):
        super().__init__()
        self._width = width
        self.stroke = stroke
        self.thickness = thickness
        self.height = 4

    def wrap(self, availWidth, availHeight):
        self.width = self._width or availWidth
        return self.width, self.height

    def draw(self):
        self.canv.setStrokeColor(self.stroke)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 2, self.width, 2)


class QuoteBox(Flowable):
    """Citazione / esempio in riquadro soft."""

    def __init__(self, text: str, width=None):
        super().__init__()
        self.text = text
        self._width = width
        self._para = None
        self._h = 0

    def wrap(self, availWidth, availHeight):
        self.width = self._width or availWidth
        style = ParagraphStyle(
            "qbox",
            fontName="NotoSerif-Italic",
            fontSize=11,
            leading=16,
            textColor=TEAL_DEEP,
            alignment=TA_LEFT,
        )
        self._para = Paragraph(self.text, style)
        w, h = self._para.wrap(self.width - 28, availHeight)
        self._h = h + 22
        return self.width, self._h

    def draw(self):
        self.canv.setFillColor(BAND)
        self.canv.roundRect(0, 0, self.width, self._h, 6, fill=1, stroke=0)
        self.canv.setStrokeColor(TEAL)
        self.canv.setLineWidth(3)
        self.canv.line(8, 8, 8, self._h - 8)
        self._para.drawOn(self.canv, 20, 11)


def _styles():
    s = getSampleStyleSheet()
    s.add(
        ParagraphStyle(
            "CoverTitle",
            fontName="NotoSerif-Bold",
            fontSize=42,
            leading=48,
            textColor=TEAL_DEEP,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    s.add(
        ParagraphStyle(
            "CoverSub",
            fontName="NotoSans",
            fontSize=14,
            leading=20,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    s.add(
        ParagraphStyle(
            "H1",
            fontName="NotoSerif-Bold",
            fontSize=18,
            leading=24,
            textColor=TEAL_DEEP,
            spaceBefore=18,
            spaceAfter=10,
        )
    )
    s.add(
        ParagraphStyle(
            "Body",
            fontName="NotoSans",
            fontSize=11,
            leading=17,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )
    )
    s.add(
        ParagraphStyle(
            "DocBullet",
            fontName="NotoSans",
            fontSize=11,
            leading=17,
            textColor=INK,
            leftIndent=14,
            spaceAfter=6,
        )
    )
    s.add(
        ParagraphStyle(
            "Caption",
            fontName="NotoSans-Italic",
            fontSize=9.5,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    s.add(
        ParagraphStyle(
            "FooterMade",
            fontName="NotoSerif-Italic",
            fontSize=13,
            leading=18,
            textColor=AMBER,
            alignment=TA_CENTER,
        )
    )
    s.add(
        ParagraphStyle(
            "SmallCenter",
            fontName="NotoSans",
            fontSize=10,
            leading=14,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    return s


def _header_footer(canvas, doc):
    canvas.saveState()
    page_w, page_h = A4
    # carta
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    # fascia superiore sottile
    if doc.page > 1:
        canvas.setFillColor(TEAL)
        canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=1, stroke=0)
        canvas.setFillColor(white)
        canvas.setFont("NotoSans", 8)
        canvas.drawString(2 * cm, page_h - 5.2 * mm, "Makima · Guida amichevole")
        canvas.drawRightString(page_w - 2 * cm, page_h - 5.2 * mm, f"{doc.page}")
        # piede
        canvas.setFillColor(MUTED)
        canvas.setFont("NotoSerif-Italic", 8)
        canvas.drawCentredString(page_w / 2, 1.2 * cm, "made by il biagigio")
    canvas.restoreState()


def _cover_page(canvas, doc):
    canvas.saveState()
    page_w, page_h = A4
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    # fascia teal laterale sinistra
    canvas.setFillColor(TEAL_DEEP)
    canvas.rect(0, 0, 14 * mm, page_h, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(14 * mm, 0, 3 * mm, page_h, fill=1, stroke=0)
    # firma in basso
    canvas.setFillColor(AMBER)
    canvas.setFont("NotoSerif-Italic", 12)
    canvas.drawCentredString(page_w / 2 + 5 * mm, 2.4 * cm, "made by il biagigio")
    canvas.setFillColor(MUTED)
    canvas.setFont("NotoSans", 8)
    canvas.drawCentredString(page_w / 2 + 5 * mm, 1.7 * cm, "Documentazione per persone, non per macchine")
    canvas.restoreState()


def build() -> Path:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.4 * cm,
        bottomMargin=2.2 * cm,
        title="Makima — Guida amichevole",
        author="il biagigio",
        subject="Documentazione semplice di Makima",
    )

    story = []

    # ——— COPERTINA ———
    story.append(Spacer(1, 4.5 * cm))
    story.append(Paragraph("Makima", styles["CoverTitle"]))
    story.append(Rule(stroke=TEAL, thickness=1.2))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Guida amichevole", styles["CoverSub"]))
    story.append(
        Paragraph(
            "Come funziona, a cosa serve, e come usarla<br/>senza parlare di codice.",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 1.2 * cm))
    story.append(
        QuoteBox(
            "«Non inventa numeri a caso. Ti dice quanto è sicura — "
            "e quando non lo sa, te lo dice.»"
        )
    )
    story.append(PageBreak())

    # ——— 1 ———
    story.append(Paragraph("1. Cos’è Makima, in due parole", styles["H1"]))
    story.append(
        Paragraph(
            "Makima è un’assistente per chi scrive software. La puoi interrogare in "
            "italiano, come faresti con una collega: «riusciremo a rilasciare entro "
            "venerdì?», «come stanno andando i deploy?», «ricordati che preferisco "
            "fare i rilasci il martedì».",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "La differenza rispetto a una chat qualunque è questa: Makima non "
            "indovina. Usa quello che ha già visto — i tuoi fatti, la storia del "
            "progetto, i risultati passati — e ti risponde in probabilità, "
            "spiegando perché.",
            styles["Body"],
        )
    )

    # ——— 2 ———
    story.append(Paragraph("2. Che problema risolve", styles["H1"]))
    story.append(
        Paragraph(
            "Quando stimiamo a occhio («secondo me ce la facciamo»), spesso "
            "siamo troppo ottimisti. Quando chiediamo a un chatbot generico, "
            "a volte inventa cifre che suonano bene ma non poggiano su nulla.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Makima sta in mezzo: ti dà una stima onesta, basata su cose "
            "davvero osservate, e ammette i limiti quando i dati mancano.",
            styles["Body"],
        )
    )

    # ——— 3 ———
    story.append(Paragraph("3. Come le parli", styles["H1"]))
    story.append(
        Paragraph(
            "Puoi usarla da terminale, da una piccola applicazione a finestra, "
            "o chiedendole di spiegare e ricordare. In pratica hai quattro modi "
            "semplici di interagire:",
            styles["Body"],
        )
    )
    bullets = [
        "<b>Chiedi una previsione</b> — tipo «qual è la probabilità di rilasciare entro dicembre?»",
        "<b>Fatti spiegare</b> — Makima ridice i numeri con parole chiare, senza inventarne di nuovi.",
        "<b>Raccontale qualcosa</b> — «preferisco i deploy il martedì» entra nella sua memoria.",
        "<b>Chiàcchiera</b> — ti risponde usando ciò che già sa di te e del progetto.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["DocBullet"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(
        QuoteBox(
            "Esempio: «Rilasceremo la nuova feature questa settimana?» "
            "— Makima guarda quanto spesso, in passato, le feature sono "
            "andate in porto, e ti risponde di conseguenza."
        )
    )

    # ——— 4 ———
    story.append(Paragraph("4. Come ricorda", styles["H1"]))
    story.append(
        Paragraph(
            "Makima tiene un diario. Se le dici un fatto importante («lavoro "
            "meglio al mattino», «questa settimana il focus è il front-end»), "
            "lo conserva e può richiamarlo più tardi quando torna utile.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Non è una memoria magica infinita: è più vicina a un taccuino "
            "ordinato. Tu scrivi, lei associa, e quando chiedi «quando "
            "facciamo il deploy?» può ripescare ciò che le avevi detto sul "
            "deploy.",
            styles["Body"],
        )
    )

    # ——— 5 ———
    story.append(Paragraph("5. Guarda anche il lavoro sul codice", styles["H1"]))
    story.append(
        Paragraph(
            "Se lo attivi, Makima può osservare la storia dei commit del "
            "progetto: quanti sono stati di nuove funzioni, quante correzioni, "
            "quanto spesso si aggiungono test. Da lì ricava abitudini reali "
            "del team, non impressioni.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "In pratica: più il progetto “vive” nel tempo, più le sue stime "
            "diventano calibrate su di te — non su un modello generico di Internet.",
            styles["Body"],
        )
    )

    story.append(PageBreak())

    # ——— 6 ———
    story.append(Paragraph("6. Cosa puoi chiederle (e cosa no)", styles["H1"]))
    story.append(Paragraph("Va bene chiedere cose come:", styles["Body"]))
    for b in [
        "Probabilità di un rilascio entro una data",
        "Quanto è stabile un pezzo del lavoro (deploy, test, feature)",
        "Un riassunto onesto di una situazione, con i numeri che ha",
        "Di ricordare preferenze e abitudini del team",
    ]:
        story.append(Paragraph(f"• {b}", styles["DocBullet"]))

    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Non è pensata per:", styles["Body"]))
    for b in [
        "Inventare certezze quando non ci sono dati",
        "Sostituire il giudizio umano sulle priorità di prodotto",
        "Raccontare barzellette o fare da chatbot generico “da salotto”",
    ]:
        story.append(Paragraph(f"• {b}", styles["DocBullet"]))

    story.append(
        Paragraph(
            "Quando non capisce o non ha abbastanza prove, preferisce dirti "
            "«non lo so con sicurezza» piuttosto che riempire il vuoto.",
            styles["Body"],
        )
    )

    # ——— 7 ———
    story.append(Paragraph("7. La finestra sul desktop", styles["H1"]))
    story.append(
        Paragraph(
            "C’è anche un’interfaccia grafica: una chat con modalità chiare "
            "(spiega, chatta, ricorda, pensa) e qualche guida del tipo "
            "«qui fai questa cosa». L’idea è che non debba indovinare dove "
            "cliccare: ogni angolo ha un lavoro preciso.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Se preferisci la riga di comando, puoi fare le stesse cose da lì. "
            "Il cuore resta lo stesso.",
            styles["Body"],
        )
    )

    # ——— 8 ———
    story.append(Paragraph("8. In sintesi", styles["H1"]))
    story.append(
        Paragraph(
            "Makima è un’assistente di previsione e memoria per chi sviluppa: "
            "parla italiano, non allucina cifre, ricorda ciò che le racconti, "
            "impara dalle abitudini del repository, e ti dice quanto è sicura "
            "di ciò che afferma.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Se vuoi una collega che calcola con calma invece di convincerti "
            "a parole — ecco, è fatta per quello.",
            styles["Body"],
        )
    )

    story.append(Spacer(1, 1.2 * cm))
    story.append(Rule(stroke=AMBER, thickness=1.0))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("made by il biagigio", styles["FooterMade"]))
    story.append(
        Paragraph(
            "Con affetto per chi preferisce capire prima di configurare.",
            styles["SmallCenter"],
        )
    )

    def first_page(canvas, doc):
        _cover_page(canvas, doc)

    def later_pages(canvas, doc):
        _header_footer(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Generato: {path}")
